package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.autograd.NormGradFunctions;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * {@link GpuNormGradOps} on each GPU API, and the {@code CpuNorm} backward references through {@link NormOps} on the
 * CPU reference backend, against double-precision references of the textbook normalization gradients (reduction
 * class of the {@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}, K the longest sum a gradient depends
 * on). Also checks the recording of the {@link NormOps} forwards: one node routing the primitive's gradients to every
 * leaf, no second running-statistic update in a training-mode BatchNorm backward, and an evaluation-mode backward that
 * keeps the statistics of its forward after the running buffers change.
 */
@RunWith(Parameterized.class)
public class GpuNormGradOpsParityTest {

    private static final float EPS = 1e-5f;

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> data() {
        return TrainParity.apis();
    }

    private final TrainParity.Api api;
    private TrainParity.Session session;
    private ComputeBackend gpu;
    private ComputeBackend host;
    private final List<Tensor> owned = new ArrayList<>();

    public GpuNormGradOpsParityTest(TrainParity.Api api) {
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
            for (Tensor t : owned) {
                t.close();
            }
            owned.clear();
        } finally {
            try {
                if (host != null) {
                    TrainGpu.release(host);
                    host.close();
                }
            } finally {
                if (session != null) {
                    session.close();
                }
            }
        }
    }

    private Tensor keep(Tensor t) {
        if (t != null) {
            owned.add(t);
        }
        return t;
    }

    private NormOps.Gradients keep(NormOps.Gradients g) {
        assertNotNull("the dispatcher must run when the train library is available", g);
        keep(g.input());
        keep(g.weight());
        keep(g.bias());
        return g;
    }

    private Tensor upload(ComputeBackend b, float[] data, int... dims) {
        Tensor flat = keep(Tensor.fromArray(b, data));
        return keep(flat.reshape(Shape.of(dims)));
    }

    private Tensor leaf(ComputeBackend b, float[] data, int... dims) {
        return keep(upload(b, data, dims).withRequiresGrad(true));
    }

    // ------------------------------------------------------------------ row normalizations

    @Test
    public void layerNormBackwardMatchesReference() {
        int rows = 7;
        int cols = 300;
        float[] x = TrainParity.uniform(201, rows * cols, -3f, 5f);
        float[] gy = TrainParity.uniform(202, rows * cols, -1f, 1f);
        float[] gamma = TrainParity.uniform(203, cols, 0.5f, 1.5f);
        Ref ref = Ref.of(x, gy, gamma, Layout.rows(rows, cols), Stats.fromData(x, Layout.rows(rows, cols), false, EPS),
                Mode.CENTERED);
        for (ComputeBackend b : new ComputeBackend[]{gpu, host}) {
            NormOps.Gradients g = keep(b == gpu
                    ? GpuNormGradOps.layerNormBackward(upload(b, gy, rows, cols), upload(b, x, rows, cols), cols,
                    upload(b, gamma, cols), EPS, true, true, Shape.vector(cols))
                    : NormOps.layerNormBackward(upload(b, gy, rows, cols), upload(b, x, rows, cols), cols,
                    upload(b, gamma, cols), EPS, true, true, Shape.vector(cols)));
            ref.assertAll("layerNorm backward on " + b.getClass().getSimpleName(), g);
            assertArrayEquals(new int[]{rows, cols}, g.input().shape().dims());
        }
    }

    @Test
    public void layerNormBackwardChunksAffineColumnsAndSkipsUnneededGradients() {
        int rows = 5000;
        int cols = 6;
        float[] x = TrainParity.uniform(211, rows * cols, -2f, 2f);
        float[] gy = TrainParity.uniform(212, rows * cols, -1f, 1f);
        float[] gamma = TrainParity.uniform(213, cols, 0.5f, 1.5f);
        Layout layout = Layout.rows(rows, cols);
        Ref ref = Ref.of(x, gy, gamma, layout, Stats.fromData(x, layout, false, EPS), Mode.CENTERED);
        Tensor gyT = upload(gpu, gy, rows, cols);
        Tensor xT = upload(gpu, x, rows, cols);
        Tensor gammaT = upload(gpu, gamma, 1, cols);

        NormOps.Gradients affineOnly = keep(GpuNormGradOps.layerNormBackward(gyT, xT, cols, gammaT, EPS, false, true,
                Shape.vector(cols)));
        assertNull("gx was not requested", affineOnly.input());
        assertArrayEquals("the gamma gradient takes gamma's shape", new int[]{1, cols},
                affineOnly.weight().shape().dims());
        ref.assertWeight("chunked gamma gradient", affineOnly.weight().toFloatArray());
        ref.assertBias("chunked fused beta gradient", affineOnly.bias().toFloatArray());

        NormOps.Gradients betaOnly = keep(GpuNormGradOps.layerNormBackward(gyT, xT, cols, null, EPS, false, false,
                Shape.vector(cols)));
        assertNull(betaOnly.input());
        assertNull(betaOnly.weight());
        TrainParity.assertBitIdentical("column sums equal the fused beta plane", affineOnly.bias().toFloatArray(),
                betaOnly.bias().toFloatArray());
    }

    @Test
    public void layerNormBackwardStagesStridedOperands() {
        int rows = 5;
        int cols = 70;
        float[] stored = TrainParity.uniform(221, rows * cols, -2f, 2f);
        float[] gy = TrainParity.uniform(222, rows * cols, -1f, 1f);
        Tensor transposed = keep(upload(gpu, stored, cols, rows).transpose());
        float[] logical = new float[rows * cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                logical[r * cols + c] = stored[c * rows + r];
            }
        }
        Layout layout = Layout.rows(rows, cols);
        Ref ref = Ref.of(logical, gy, null, layout, Stats.fromData(logical, layout, false, EPS), Mode.CENTERED);
        NormOps.Gradients g = keep(GpuNormGradOps.layerNormBackward(upload(gpu, gy, rows, cols), transposed, cols,
                null, EPS, true, false, null));
        ref.assertInput("layerNorm backward of a transposed view", g.input().toFloatArray());
    }

    @Test
    public void rmsNormBackwardMatchesReference() {
        int rows = 9;
        int cols = 40;
        float eps = 1e-8f;
        float[] x = TrainParity.uniform(231, rows * cols, -4f, 4f);
        float[] gy = TrainParity.uniform(232, rows * cols, -1f, 1f);
        float[] w = TrainParity.uniform(233, cols, 0.25f, 2f);
        Layout layout = Layout.rows(rows, cols);
        Ref ref = Ref.of(x, gy, w, layout, Stats.fromData(x, layout, true, eps), Mode.RMS);
        for (ComputeBackend b : new ComputeBackend[]{gpu, host}) {
            NormOps.Gradients g = keep(b == gpu
                    ? GpuNormGradOps.rmsNormBackward(upload(b, gy, 3, 3, cols), upload(b, x, 3, 3, cols),
                    upload(b, w, cols), eps, true, true)
                    : NormOps.rmsNormBackward(upload(b, gy, 3, 3, cols), upload(b, x, 3, 3, cols),
                    upload(b, w, cols), eps, true, true));
            assertNull(g.bias());
            ref.assertAll("rmsNorm backward on " + b.getClass().getSimpleName(), g);
        }
    }

    // ------------------------------------------------------------------ channel normalizations

    @Test
    public void batchNormTrainBackwardMatchesReference() {
        int b = 3;
        int c = 4;
        int hw = 6 * 5;
        float[] all = TrainParity.uniform(241, (b + 1) * c * hw, -2f, 3f);
        float[] x = new float[b * c * hw];
        System.arraycopy(all, c * hw, x, 0, x.length);
        float[] gy = TrainParity.uniform(242, b * c * hw, -1f, 1f);
        float[] gamma = TrainParity.uniform(243, c, 0.5f, 1.5f);
        Layout layout = Layout.channels(b, c, hw, 0);
        Ref ref = Ref.of(x, gy, gamma, layout, Stats.fromData(x, layout, false, EPS), Mode.CENTERED);

        Tensor sliced = keep(upload(gpu, all, b + 1, c, 6, 5).slice(0, 1, b + 1));
        NormOps.Gradients device = keep(GpuNormGradOps.batchNormBackward(upload(gpu, gy, b, c, 6, 5), sliced,
                upload(gpu, gamma, c), null, null, true, EPS, true, true, Shape.vector(c)));
        ref.assertAll("batchNorm train backward on the device", device);

        NormOps.Gradients cpu = keep(NormOps.batchNormBackward(upload(host, gy, b, c, 6, 5),
                upload(host, x, b, c, 6, 5), upload(host, gamma, c), null, null, true, EPS, true, true,
                Shape.vector(c)));
        ref.assertAll("batchNorm train backward on the CPU path", cpu);
    }

    @Test
    public void batchNormEvalBackwardMatchesReference() {
        int b = 2;
        int c = 3;
        int l = 17;
        float[] x = TrainParity.uniform(251, b * c * l, -2f, 2f);
        float[] gy = TrainParity.uniform(252, b * c * l, -1f, 1f);
        float[] mean = TrainParity.uniform(253, c, -0.5f, 0.5f);
        float[] var = TrainParity.uniform(254, c, 0.5f, 2f);
        float[] gamma = TrainParity.uniform(255, c, 0.5f, 1.5f);
        Layout layout = Layout.channels(b, c, l, 0);
        Ref ref = Ref.of(x, gy, gamma, layout, Stats.given(mean, var, EPS), Mode.CONSTANT);
        for (ComputeBackend be : new ComputeBackend[]{gpu, host}) {
            NormOps.Gradients g = keep(be == gpu
                    ? GpuNormGradOps.batchNormBackward(upload(be, gy, b, c, l), upload(be, x, b, c, l),
                    upload(be, gamma, c), upload(be, mean, c), upload(be, var, c), false, EPS, true, true,
                    Shape.vector(c))
                    : NormOps.batchNormBackward(upload(be, gy, b, c, l), upload(be, x, b, c, l),
                    upload(be, gamma, c), upload(be, mean, c), upload(be, var, c), false, EPS, true, true,
                    Shape.vector(c)));
            ref.assertAll("batchNorm eval backward on " + be.getClass().getSimpleName(), g);
        }
    }

    @Test
    public void groupNormBackwardMatchesReference() {
        int b = 2;
        int c = 6;
        int s = 4 * 3;
        int groups = 3;
        float[] x = TrainParity.uniform(261, b * c * s, -3f, 3f);
        float[] gy = TrainParity.uniform(262, b * c * s, -1f, 1f);
        float[] gamma = TrainParity.uniform(263, c, 0.5f, 1.5f);
        Layout layout = Layout.channels(b, c, s, groups);
        Ref ref = Ref.of(x, gy, gamma, layout, Stats.fromData(x, layout, false, EPS), Mode.CENTERED);
        for (ComputeBackend be : new ComputeBackend[]{gpu, host}) {
            NormOps.Gradients g = keep(be == gpu
                    ? GpuNormGradOps.groupNormBackward(upload(be, gy, b, c, 4, 3), upload(be, x, b, c, 4, 3),
                    groups, upload(be, gamma, c), EPS, true, true, Shape.vector(c))
                    : NormOps.groupNormBackward(upload(be, gy, b, c, 4, 3), upload(be, x, b, c, 4, 3), groups,
                    upload(be, gamma, c), EPS, true, true, Shape.vector(c)));
            ref.assertAll("groupNorm backward on " + be.getClass().getSimpleName(), g);
        }
    }

    @Test
    public void emptyBatchGivesZeroAffineGradients() {
        Shape empty = Shape.of(0, 3, 4);
        Tensor x = keep(Tensor.zeros(gpu, empty));
        Tensor gy = keep(Tensor.zeros(gpu, empty));
        Tensor gamma = upload(gpu, new float[]{1f, 2f, 3f}, 3);
        NormOps.Gradients bn = keep(GpuNormGradOps.batchNormBackward(gy, x, gamma, null, null, true, EPS, true, true,
                Shape.vector(3)));
        assertEquals(0, bn.input().numElements());
        TrainParity.assertBitIdentical("batchNorm gamma gradient", new float[3], bn.weight().toFloatArray());
        TrainParity.assertBitIdentical("batchNorm beta gradient", new float[3], bn.bias().toFloatArray());
        NormOps.Gradients gn = keep(GpuNormGradOps.groupNormBackward(gy, x, 3, gamma, EPS, true, true,
                Shape.vector(3)));
        TrainParity.assertBitIdentical("groupNorm gamma gradient", new float[3], gn.weight().toFloatArray());
        TrainParity.assertBitIdentical("groupNorm beta gradient", new float[3], gn.bias().toFloatArray());
    }

    // ------------------------------------------------------------------ recording

    @Test
    public void recordedLayerNormRoutesThePrimitiveGradients() {
        int rows = 4;
        int cols = 33;
        float[] xd = TrainParity.uniform(271, rows * cols, -2f, 2f);
        float[] gyd = TrainParity.uniform(272, rows * cols, -1f, 1f);
        float[] gd = TrainParity.uniform(273, cols, 0.5f, 1.5f);
        float[] bd = TrainParity.uniform(274, cols, -0.5f, 0.5f);
        Tensor x = leaf(gpu, xd, rows, cols);
        Tensor gamma = leaf(gpu, gd, cols);
        Tensor beta = leaf(gpu, bd, cols);
        Tensor gy = upload(gpu, gyd, rows, cols);
        int before = ComputationGraph.current().size();

        Tensor out = keep(NormOps.layerNorm(x, cols, gamma, beta, EPS));
        assertTrue(out.requiresGrad());
        assertTrue(out.gradFn() instanceof NormGradFunctions.LayerNormGrad);
        assertEquals("one node per call", before + 1, ComputationGraph.current().size());

        NormOps.Gradients expected = keep(NormOps.layerNormBackward(gy, x, cols, gamma, EPS, true, true,
                beta.shape()));
        ComputationGraph.current().backward(out, gy);
        assertEquals("the backward released the node", before, ComputationGraph.current().size());
        TrainParity.assertBitIdentical("x gradient", expected.input().toFloatArray(), keep(x.grad()).toFloatArray());
        TrainParity.assertBitIdentical("gamma gradient", expected.weight().toFloatArray(),
                keep(gamma.grad()).toFloatArray());
        TrainParity.assertBitIdentical("beta gradient", expected.bias().toFloatArray(),
                keep(beta.grad()).toFloatArray());
    }

    @Test
    public void unrecordedNormalizationsDoNotRequireGrad() {
        Tensor x = leaf(gpu, TrainParity.uniform(281, 2 * 4 * 3, -1f, 1f), 2, 4, 3);
        int before = ComputationGraph.current().size();
        try (ComputationGraph.NoGradScope off = ComputationGraph.noGrad()) {
            Tensor y = keep(NormOps.groupNorm(x, 2, null, null, EPS));
            assertTrue(!y.requiresGrad() && y.gradFn() == null);
        }
        Tensor constant = upload(gpu, TrainParity.uniform(282, 2 * 4 * 3, -1f, 1f), 2, 4, 3);
        Tensor z = keep(NormOps.rmsNorm(constant, null, EPS));
        assertTrue(!z.requiresGrad() && z.gradFn() == null);
        assertEquals(before, ComputationGraph.current().size());
    }

    @Test
    public void batchNormTrainBackwardDoesNotUpdateRunningStatsAgain() {
        int b = 3;
        int c = 2;
        int l = 11;
        float[] xd = TrainParity.uniform(291, b * c * l, -1f, 2f);
        float[] gyd = TrainParity.uniform(292, b * c * l, -1f, 1f);
        Tensor x = leaf(gpu, xd, b, c, l);
        Tensor gamma = leaf(gpu, new float[]{0.75f, 1.25f}, c);
        Tensor beta = leaf(gpu, new float[]{0.1f, -0.2f}, c);
        Tensor rm = upload(gpu, new float[]{0.05f, -0.05f}, c);
        Tensor rv = upload(gpu, new float[]{1.0f, 0.9f}, c);
        Tensor gy = upload(gpu, gyd, b, c, l);

        Tensor out = keep(NormOps.batchNorm(x, gamma, beta, rm, rv, true, 0.1f, EPS));
        assertTrue(out.gradFn() instanceof NormGradFunctions.BatchNormGrad);
        float[] meanAfterForward = rm.toFloatArray();
        float[] varAfterForward = rv.toFloatArray();
        NormOps.Gradients expected = keep(NormOps.batchNormBackward(gy, x, gamma, null, null, true, EPS, true, true,
                beta.shape()));
        ComputationGraph.current().backward(out, gy);

        TrainParity.assertBitIdentical("running mean after backward", meanAfterForward, rm.toFloatArray());
        TrainParity.assertBitIdentical("running var after backward", varAfterForward, rv.toFloatArray());
        TrainParity.assertBitIdentical("x gradient", expected.input().toFloatArray(), keep(x.grad()).toFloatArray());
        TrainParity.assertBitIdentical("gamma gradient", expected.weight().toFloatArray(),
                keep(gamma.grad()).toFloatArray());
        TrainParity.assertBitIdentical("beta gradient", expected.bias().toFloatArray(),
                keep(beta.grad()).toFloatArray());
    }

    @Test
    public void batchNormEvalBackwardKeepsTheForwardStatistics() {
        int b = 2;
        int c = 3;
        int l = 8;
        float[] xd = TrainParity.uniform(301, b * c * l, -2f, 2f);
        float[] gyd = TrainParity.uniform(302, b * c * l, -1f, 1f);
        Tensor x = leaf(gpu, xd, b, c, l);
        Tensor gamma = leaf(gpu, new float[]{0.5f, 1f, 1.5f}, c);
        Tensor beta = leaf(gpu, new float[]{0f, 0.25f, -0.25f}, c);
        Tensor rm = upload(gpu, new float[]{0.1f, -0.2f, 0.3f}, c);
        Tensor rv = upload(gpu, new float[]{0.8f, 1.1f, 1.4f}, c);
        Tensor gy = upload(gpu, gyd, b, c, l);

        Tensor out = keep(NormOps.batchNorm(x, gamma, beta, rm, rv, false, 0.1f, EPS));
        NormOps.Gradients expected = keep(NormOps.batchNormBackward(gy, x, gamma, rm, rv, false, EPS, true, true,
                beta.shape()));
        try (ComputationGraph.NoGradScope off = ComputationGraph.noGrad()) {
            Tensor other = upload(gpu, TrainParity.uniform(303, b * c * l, 3f, 9f), b, c, l);
            keep(NormOps.batchNorm(other, null, null, rm, rv, true, 0.5f, EPS));
        }
        ComputationGraph.current().backward(out, gy);

        TrainParity.assertBitIdentical("x gradient uses the forward's statistics", expected.input().toFloatArray(),
                keep(x.grad()).toFloatArray());
        TrainParity.assertBitIdentical("gamma gradient uses the forward's statistics",
                expected.weight().toFloatArray(), keep(gamma.grad()).toFloatArray());
        TrainParity.assertBitIdentical("beta gradient", expected.bias().toFloatArray(),
                keep(beta.grad()).toFloatArray());
    }

    // ------------------------------------------------------------------ CPU path

    @Test
    public void everyBackwardDeclinesWithoutTrainLibrary() {
        float[] x = TrainParity.uniform(311, 2 * 4 * 3, -1f, 1f);
        Tensor in = upload(host, x, 2, 4, 3);
        Tensor gy = upload(host, x, 2, 4, 3);
        Tensor stats = upload(host, new float[]{0.1f, 0.2f, 0.3f, 0.4f}, 4);
        assertNull(GpuNormGradOps.layerNormBackward(gy, in, 3, null, EPS, true, false, null));
        assertNull(GpuNormGradOps.rmsNormBackward(gy, in, null, EPS, true, false));
        assertNull(GpuNormGradOps.batchNormBackward(gy, in, null, null, null, true, EPS, true, false, null));
        assertNull(GpuNormGradOps.batchNormBackward(gy, in, null, stats, stats, false, EPS, true, false, null));
        assertNull(GpuNormGradOps.groupNormBackward(gy, in, 2, null, EPS, true, false, null));
    }

    // ------------------------------------------------------------------ references

    /** How each element maps to its normalization set and to its affine parameter. */
    private static final class Layout {
        final int[] set;
        final int sets;
        final int[] param;
        final int params;

        private Layout(int[] set, int sets, int[] param, int params) {
            this.set = set;
            this.sets = sets;
            this.param = param;
            this.params = params;
        }

        /** Rows of {@code cols} elements; the parameter index is the column. */
        static Layout rows(int rows, int cols) {
            int[] set = new int[rows * cols];
            int[] param = new int[rows * cols];
            for (int e = 0; e < set.length; e++) {
                set[e] = e / cols;
                param[e] = e % cols;
            }
            return new Layout(set, rows, param, cols);
        }

        /** {@code [b, c, s]}: per-channel sets ({@code groups == 0}) or (batch, group) sets; parameter: channel. */
        static Layout channels(int b, int c, int s, int groups) {
            int[] set = new int[b * c * s];
            int[] param = new int[b * c * s];
            for (int e = 0; e < set.length; e++) {
                int ch = (e / s) % c;
                int bi = e / (s * c);
                set[e] = groups == 0 ? ch : bi * groups + ch / (c / groups);
                param[e] = ch;
            }
            return new Layout(set, groups == 0 ? c : b * groups, param, c);
        }
    }

    /** Double-precision center and scale of each set. */
    private static final class Stats {
        final double[] center;
        final double[] sdev;

        private Stats(double[] center, double[] sdev) {
            this.center = center;
            this.sdev = sdev;
        }

        static Stats fromData(float[] x, Layout l, boolean rms, float eps) {
            double[] sum = new double[l.sets];
            double[] sq = new double[l.sets];
            int[] n = new int[l.sets];
            for (int e = 0; e < x.length; e++) {
                sum[l.set[e]] += x[e];
                sq[l.set[e]] += (double) x[e] * x[e];
                n[l.set[e]]++;
            }
            double[] center = new double[l.sets];
            double[] sdev = new double[l.sets];
            double[] var = new double[l.sets];
            for (int k = 0; k < l.sets; k++) {
                center[k] = rms ? 0 : sum[k] / n[k];
            }
            for (int e = 0; e < x.length; e++) {
                double dv = x[e] - center[l.set[e]];
                var[l.set[e]] += dv * dv;
            }
            for (int k = 0; k < l.sets; k++) {
                sdev[k] = rms ? Math.sqrt(sq[k] / n[k] + eps) : Math.sqrt(var[k] / n[k] + eps);
            }
            return new Stats(center, sdev);
        }

        static Stats given(float[] mean, float[] var, float eps) {
            double[] center = new double[mean.length];
            double[] sdev = new double[mean.length];
            for (int k = 0; k < mean.length; k++) {
                center[k] = mean[k];
                sdev[k] = Math.sqrt((double) var[k] + eps);
            }
            return new Stats(center, sdev);
        }
    }

    /**
     * The gradient formula: CENTERED {@code gx = (d - mean(d) - xhat * mean(d * xhat)) / sdev} (layer, batch
     * training, group), RMS {@code gx = (d - xhat * mean(d * xhat)) / rms}, CONSTANT statistics
     * {@code gx = d / sdev} (batch evaluation).
     */
    private enum Mode { CENTERED, RMS, CONSTANT }

    /** Double-precision gradients with the absolute term sums and reduction lengths of their tolerance bounds. */
    private static final class Ref {
        final double[] gx;
        final double[] gxTerms;
        final int[] gxK;
        final double[] gw;
        final double[] gwTerms;
        final double[] gb;
        final double[] gbTerms;
        final int[] paramK;

        private Ref(int n, int params) {
            gx = new double[n];
            gxTerms = new double[n];
            gxK = new int[n];
            gw = new double[params];
            gwTerms = new double[params];
            gb = new double[params];
            gbTerms = new double[params];
            paramK = new int[params];
        }

        static Ref of(float[] x, float[] gy, float[] scale, Layout l, Stats st, Mode mode) {
            Ref r = new Ref(x.length, l.params);
            double[] sumD = new double[l.sets];
            double[] sumDX = new double[l.sets];
            double[] absD = new double[l.sets];
            double[] absDX = new double[l.sets];
            int[] n = new int[l.sets];
            for (int e = 0; e < x.length; e++) {
                int k = l.set[e];
                double sd = st.sdev[k];
                double xhat = (x[e] - st.center[k]) / sd;
                double xa = (Math.abs(x[e]) + Math.abs(st.center[k])) / sd;
                double d = gy[e] * (scale == null ? 1.0 : scale[l.param[e]]);
                sumD[k] += d;
                sumDX[k] += d * xhat;
                absD[k] += Math.abs(d);
                absDX[k] += Math.abs(d) * xa;
                n[k]++;
                int p = l.param[e];
                r.gw[p] += gy[e] * xhat;
                r.gwTerms[p] += Math.abs(gy[e]) * xa;
                r.gb[p] += gy[e];
                r.gbTerms[p] += Math.abs(gy[e]);
                r.paramK[p]++;
            }
            for (int e = 0; e < x.length; e++) {
                int k = l.set[e];
                double sd = st.sdev[k];
                double xhat = (x[e] - st.center[k]) / sd;
                double xa = (Math.abs(x[e]) + Math.abs(st.center[k])) / sd;
                double d = gy[e] * (scale == null ? 1.0 : scale[l.param[e]]);
                switch (mode) {
                    case CENTERED:
                        r.gx[e] = (d - sumD[k] / n[k] - xhat * (sumDX[k] / n[k])) / sd;
                        r.gxTerms[e] = (Math.abs(d) + absD[k] / n[k] + xa * (absDX[k] / n[k])) / sd;
                        r.gxK[e] = n[k];
                        break;
                    case RMS:
                        r.gx[e] = (d - xhat * (sumDX[k] / n[k])) / sd;
                        r.gxTerms[e] = (Math.abs(d) + xa * (absDX[k] / n[k])) / sd;
                        r.gxK[e] = n[k];
                        break;
                    default:
                        r.gx[e] = d / sd;
                        r.gxTerms[e] = Math.abs(d) / sd;
                        r.gxK[e] = 1;
                        break;
                }
            }
            return r;
        }

        void assertAll(String what, NormOps.Gradients g) {
            assertInput(what + " gx", g.input().toFloatArray());
            assertWeight(what + " gamma", g.weight().toFloatArray());
            if (g.bias() != null) {
                assertBias(what + " beta", g.bias().toFloatArray());
            }
        }

        void assertInput(String what, float[] got) {
            assertEquals(what + ": length", gx.length, got.length);
            for (int e = 0; e < got.length; e++) {
                TrainParity.assertReduction(what + " [" + e + "]", gx[e], got[e], gxK[e], gxTerms[e]);
            }
        }

        void assertWeight(String what, float[] got) {
            assertEquals(what + ": length", gw.length, got.length);
            for (int p = 0; p < got.length; p++) {
                TrainParity.assertReduction(what + " [" + p + "]", gw[p], got[p], Math.max(1, paramK[p]),
                        gwTerms[p]);
            }
        }

        void assertBias(String what, float[] got) {
            assertEquals(what + ": length", gb.length, got.length);
            for (int p = 0; p < got.length; p++) {
                TrainParity.assertReduction(what + " [" + p + "]", gb[p], got[p], Math.max(1, paramK[p]),
                        gbTerms[p]);
            }
        }
    }
}

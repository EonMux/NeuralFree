package com.eonmux.neuralfree.ml;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
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
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * {@link Perceptron} on the CPU reference path (a host backend without train kernels) and on every GPU API with the
 * train kernels available (skipped elsewhere): online learning behaviour, parameter round trips, input validation,
 * and parity of the forward pass and of one SGD step with a double-precision reference under the tolerance classes
 * of the {@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}.
 */
@RunWith(Parameterized.class)
public class PerceptronTest {

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> backends() {
        return LinearReference.backends();
    }

    private final TrainParity.Api api;
    private TrainParity.Session session;
    private ComputeBackend backend;

    public PerceptronTest(String label, TrainParity.Api api) {
        this.api = api;
    }

    @Before
    public void setUp() {
        if (api == null) {
            backend = TrainParity.hostBackend();
        } else {
            session = TrainParity.open(api);
            backend = session.backend();
        }
    }

    @After
    public void tearDown() {
        if (session != null) {
            session.close();
        }
    }

    @Test
    public void learnsALinearlySeparableRuleOnline() {
        try (Perceptron p = new Perceptron(backend, 2, 0.5f, 0.0f)) {
            Random rng = new Random(42);
            for (int step = 0; step < 300; step++) {
                float[][] x = new float[16][2];
                float[] y = new float[16];
                for (int i = 0; i < 16; i++) {
                    x[i][0] = rng.nextFloat() * 2 - 1;
                    x[i][1] = rng.nextFloat() * 2 - 1;
                    y[i] = x[i][0] > x[i][1] ? 1.0f : 0.0f;
                }
                p.update(x, y);
            }
            assertEquals(300L * 16, p.samplesSeen());

            int correct = 0;
            int total = 500;
            for (int i = 0; i < total; i++) {
                float a = rng.nextFloat() * 2 - 1;
                float b = rng.nextFloat() * 2 - 1;
                boolean predicted = p.predictOne(new float[]{a, b}) >= 0.5f;
                if (predicted == (a > b)) {
                    correct++;
                }
            }
            assertTrue("online SGD must learn x0 > x1 (got " + correct + "/" + total + ")",
                    correct >= (int) (total * 0.95));
        }
    }

    @Test
    public void parametersRoundTripAndDriveTheForwardPass() {
        try (Perceptron p = new Perceptron(backend, 3)) {
            float[] w = {0.5f, -1.25f, 2.0f};
            p.setParameters(w, 0.75f);
            TrainParity.assertBitIdentical("weights", w, p.weights());
            assertEquals(Float.floatToRawIntBits(0.75f), Float.floatToRawIntBits(p.bias()));

            float[] x = {1.0f, 2.0f, -0.5f};
            double z = 0.75 + 0.5 * 1.0 + (-1.25) * 2.0 + 2.0 * (-0.5);
            assertEquals(1.0 / (1.0 + Math.exp(-z)), p.predictOne(x), 1e-5);
        }
    }

    @Test
    public void startsFromZeroParameters() {
        try (Perceptron p = new Perceptron(backend, 4)) {
            TrainParity.assertBitIdentical("weights", new float[4], p.weights());
            assertEquals(0, Float.floatToRawIntBits(p.bias()));
            assertEquals(0.5f, p.predictOne(new float[]{1f, -2f, 3f, -4f}), 1e-6f);
        }
    }

    @Test
    public void emptyBatchesNeedNoWork() {
        try (Perceptron p = new Perceptron(backend, 2)) {
            assertEquals(0, p.predict(new float[0][]).length);
            p.update(new float[0][], new float[0]);
            assertEquals(0L, p.samplesSeen());
        }
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsSamplesWithTheWrongFeatureCount() {
        try (Perceptron p = new Perceptron(backend, 4)) {
            p.predictOne(new float[]{1, 2, 3});
        }
    }

    /**
     * For one repeated feature vector labelled 1 at weight 3 and 0 at weight 1, the weighted logistic MLE is the
     * weighted label mean 3/4, while the unweighted case settles at the plain mean 1/2.
     */
    @Test
    public void sampleWeightsScaleEachSamplesInfluence() {
        float[][] x = {{1.0f}, {1.0f}};
        float[] y = {1.0f, 0.0f};
        try (Perceptron weighted = new Perceptron(backend, 1, 0.5f, 0.0f);
             Perceptron unweighted = new Perceptron(backend, 1, 0.5f, 0.0f)) {
            for (int e = 0; e < 400; e++) {
                weighted.update(x, y, new float[]{3.0f, 1.0f});
                unweighted.update(x, y);
            }
            assertEquals(0.75, weighted.predictOne(new float[]{1.0f}), 0.02);
            assertEquals(0.5, unweighted.predictOne(new float[]{1.0f}), 0.02);
        }
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsNegativeSampleWeights() {
        try (Perceptron p = new Perceptron(backend, 1)) {
            p.update(new float[][]{{1.0f}}, new float[]{1.0f}, new float[]{-1.0f});
        }
    }

    @Test(expected = IllegalStateException.class)
    public void rejectsUseAfterClose() {
        Perceptron p = new Perceptron(backend, 1);
        p.close();
        p.close();
        p.weights();
    }

    /**
     * Releasing the backend's train context (as the dispatchers' {@code clearCache} does) leaves a model usable: its
     * next step reopens the context and computes exactly what a model created afterwards computes.
     */
    @Test
    public void keepsWorkingAfterTheTrainContextIsReleased() {
        float[] w = {0.25f, -0.5f};
        float[][] x = {{1.0f, 2.0f}, {-0.5f, 0.75f}};
        float[] y = {1.0f, 0.0f};
        try (Perceptron before = new Perceptron(backend, 2, 0.5f, 0.01f)) {
            before.setParameters(w, 0.125f);
            TrainGpu.release(backend);
            before.update(x, y);
            try (Perceptron after = new Perceptron(backend, 2, 0.5f, 0.01f)) {
                after.setParameters(w, 0.125f);
                after.update(x, y);
                TrainParity.assertBitIdentical("weights", after.weights(), before.weights());
                assertEquals(Float.floatToRawIntBits(after.bias()), Float.floatToRawIntBits(before.bias()));
                TrainParity.assertBitIdentical("predictions", after.predict(x), before.predict(x));
            }
        }
    }

    /** Many samples (several forward work-groups) and many features (several gradient work-groups). */
    @Test
    public void forwardMatchesTheDoubleReference() {
        checkForward(300, 7, 11);
        checkForward(19, 290, 12);
    }

    @Test
    public void oneWeightedStepMatchesTheDoubleReference() {
        checkStep(300, 7, 21, true);
        checkStep(19, 290, 22, true);
    }

    @Test
    public void oneUnweightedStepMatchesTheDoubleReference() {
        checkStep(300, 7, 31, false);
        checkStep(19, 290, 32, false);
    }

    private void checkForward(int n, int f, long seed) {
        float[] w = TrainParity.uniform(seed, f, -0.5f, 0.5f);
        float b = TrainParity.uniform(seed + 1, 1, -0.5f, 0.5f)[0];
        float[][] x = LinearReference.batch(seed + 2, n, f);
        try (Perceptron p = new Perceptron(backend, f)) {
            p.setParameters(w, b);
            float[] got = p.predict(x);
            LinearReference.Unit ref = LinearReference.Unit.exact(w, b);
            float[] flat = LinearReference.flatten(x);
            for (int i = 0; i < n; i++) {
                double z = ref.z(flat, i * f);
                LinearReference.assertWithin("prediction " + i + " (" + n + "x" + f + ")",
                        LinearReference.sigmoid(z), got[i], LinearReference.sigmoidErr(ref.zErr(flat, i * f)));
            }
        }
    }

    private void checkStep(int n, int f, long seed, boolean weighted) {
        float lr = 0.3f;
        float l2 = 0.05f;
        float[] w = TrainParity.uniform(seed, f, -0.5f, 0.5f);
        float b = TrainParity.uniform(seed + 1, 1, -0.5f, 0.5f)[0];
        float[][] x = LinearReference.batch(seed + 2, n, f);
        float[] y = LinearReference.labels(seed + 3, n);
        float[] s = weighted ? TrainParity.uniform(seed + 4, n, 0f, 3f) : null;
        try (Perceptron p = new Perceptron(backend, f, lr, l2)) {
            p.setParameters(w, b);
            p.update(x, y, s);

            LinearReference.Unit ref = LinearReference.Unit.exact(w, b);
            float[] flat = LinearReference.flatten(x);
            double[] e = new double[n];
            double[] eErr = new double[n];
            for (int i = 0; i < n; i++) {
                e[i] = LinearReference.sigmoid(ref.z(flat, i * f)) - y[i];
                eErr[i] = LinearReference.sigmoidErr(ref.zErr(flat, i * f)) + 1e-7;
            }
            double[][] g = LinearReference.gradient(flat, n, f, s, e, eErr);
            ref.step(g[0], g[1], lr, l2).assertMatches("step (" + n + "x" + f + ")", p.weights(), p.bias());
        }
    }

    /**
     * Double-precision reference of the logistic and linear units, with an absolute error budget for their float
     * paths: fused sums in the reduction class, the logistic in the transcendental class (its slope is at most 1/4),
     * and each further float operation a few roundings.
     */
    static final class LinearReference {
        /** Relative budget of a handful of float roundings. */
        static final double ROUNDING = 1e-6;

        private LinearReference() {
        }

        /** Parameterized rows {@code {label, api}}: the CPU reference path (api null), then each GPU API. */
        static Collection<Object[]> backends() {
            List<Object[]> rows = new ArrayList<>();
            rows.add(new Object[]{"CPU", null});
            for (Object[] row : TrainParity.apis()) {
                rows.add(new Object[]{row[0].toString(), row[0]});
            }
            return rows;
        }

        static float[][] batch(long seed, int n, int f) {
            float[] flat = TrainParity.uniform(seed, n * f, -1f, 1f);
            float[][] x = new float[n][f];
            for (int i = 0; i < n; i++) {
                System.arraycopy(flat, i * f, x[i], 0, f);
            }
            return x;
        }

        static float[] labels(long seed, int n) {
            Random r = new Random(seed);
            float[] y = new float[n];
            for (int i = 0; i < n; i++) {
                y[i] = r.nextBoolean() ? 1f : 0f;
            }
            return y;
        }

        static float[] flatten(float[][] x) {
            int f = x.length == 0 ? 0 : x[0].length;
            float[] flat = new float[x.length * f];
            for (int i = 0; i < x.length; i++) {
                System.arraycopy(x[i], 0, flat, i * f, f);
            }
            return flat;
        }

        static double sigmoid(double z) {
            return 1.0 / (1.0 + Math.exp(-z));
        }

        /** Bound of a float logistic whose argument is within {@code zErr}: slope 1/4 plus the transcendental class. */
        static double sigmoidErr(double zErr) {
            return 0.25 * zErr + 1e-5;
        }

        /** Reduction class: {@code 1e-5 * sqrt(k) * sum|terms| + k * FLT_MIN}. */
        static double reductionBound(int k, double sumAbsTerms) {
            return 1e-5 * Math.sqrt(k) * sumAbsTerms + k * (double) TrainParity.FLT_MIN;
        }

        /**
         * Mean weighted gradient {@code {g, gErr}} of length {@code f + 1} (bias last): g[j] = sum_i s_i e_i x_ij / n,
         * g[f] = sum_i s_i e_i / n, with {@code s} null meaning all ones and {@code eErr} the bound of each error.
         */
        static double[][] gradient(float[] x, int n, int f, float[] s, double[] e, double[] eErr) {
            double[] g = new double[f + 1];
            double[] err = new double[f + 1];
            for (int j = 0; j <= f; j++) {
                double sum = 0;
                double sumAbs = 0;
                double propagated = 0;
                for (int i = 0; i < n; i++) {
                    double si = s == null ? 1.0 : s[i];
                    double xi = j == f ? 1.0 : x[i * f + j];
                    sum += si * e[i] * xi;
                    sumAbs += Math.abs(si * e[i] * xi);
                    propagated += Math.abs(si * xi) * eErr[i];
                }
                g[j] = sum / n;
                err[j] = (propagated + reductionBound(n, sumAbs)) / n + ROUNDING * Math.abs(g[j]);
            }
            return new double[][]{g, err};
        }

        static void assertWithin(String what, double ref, float got, double bound) {
            if (!(Math.abs(got - ref) <= bound)) {
                fail(what + ": got " + got + ", reference " + ref + ", bound " + bound);
            }
        }

        /** A unit's weights and bias (last) in double, with an absolute error bound for each. */
        static final class Unit {
            final double[] p;
            final double[] err;

            Unit(double[] p, double[] err) {
                this.p = p;
                this.err = err;
            }

            static Unit exact(float[] w, float b) {
                double[] p = new double[w.length + 1];
                for (int j = 0; j < w.length; j++) {
                    p[j] = w[j];
                }
                p[w.length] = b;
                return new Unit(p, new double[w.length + 1]);
            }

            int features() {
                return p.length - 1;
            }

            /** {@code b + sum_j w_j x[row + j]}. */
            double z(float[] x, int row) {
                int f = features();
                double z = p[f];
                for (int j = 0; j < f; j++) {
                    z += p[j] * x[row + j];
                }
                return z;
            }

            /** Bound of the float pre-activation: the fused sum's reduction class plus the parameters' errors. */
            double zErr(float[] x, int row) {
                int f = features();
                double sumAbs = Math.abs(p[f]);
                double propagated = err[f];
                for (int j = 0; j < f; j++) {
                    sumAbs += Math.abs(p[j] * x[row + j]);
                    propagated += Math.abs(x[row + j]) * err[j];
                }
                return reductionBound(f + 1, sumAbs) + propagated;
            }

            /** One SGD step: w - lr (g + l2 w) for the weights and b - lr g_b for the bias. */
            Unit step(double[] g, double[] gErr, float lr, float l2) {
                int f = features();
                double[] next = new double[f + 1];
                double[] nextErr = new double[f + 1];
                for (int j = 0; j <= f; j++) {
                    double decay = j == f ? 0.0 : l2;
                    next[j] = p[j] - lr * (g[j] + decay * p[j]);
                    nextErr[j] = err[j] * (1.0 + lr * decay) + lr * gErr[j]
                            + ROUNDING * (Math.abs(p[j]) + lr * (Math.abs(g[j]) + decay * Math.abs(p[j]))
                            + Math.abs(next[j]));
                }
                return new Unit(next, nextErr);
            }

            void assertMatches(String what, float[] w, float b) {
                int f = features();
                assertEquals(what + ": weight count", f, w.length);
                for (int j = 0; j < f; j++) {
                    assertWithin(what + ": weight " + j, p[j], w[j], err[j]);
                }
                assertWithin(what + ": bias", p[f], b, err[f]);
            }
        }
    }
}

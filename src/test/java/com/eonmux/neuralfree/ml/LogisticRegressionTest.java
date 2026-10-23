package com.eonmux.neuralfree.ml;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.ml.PerceptronTest.LinearReference;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Collection;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * {@link LogisticRegression} on the CPU reference path (a host backend without train kernels) and on every GPU API
 * with the train kernels available (skipped elsewhere): learning a separable rule, training iterations that take each
 * sample's features into account, the convergence stop, prediction parity, and input validation. Parity is with a
 * double-precision reference of the iterations under the tolerance classes of the
 * {@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}.
 */
@RunWith(Parameterized.class)
public class LogisticRegressionTest {

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> backends() {
        return LinearReference.backends();
    }

    private final TrainParity.Api api;
    private TrainParity.Session session;
    private ComputeBackend backend;

    public LogisticRegressionTest(String label, TrainParity.Api api) {
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
    public void learnsALinearlySeparableRule() {
        Random rng = new Random(5);
        int n = 400;
        float[][] x = new float[n][2];
        int[] y = new int[n];
        for (int i = 0; i < n; i++) {
            x[i][0] = rng.nextFloat() * 2 - 1;
            x[i][1] = rng.nextFloat() * 2 - 1;
            y[i] = x[i][0] > x[i][1] ? 1 : 0;
        }
        try (LogisticRegression lr = new LogisticRegression(backend, 1.0f, 0.0f, 500, 1e-6f)) {
            lr.fit(x, y);
            int[] predicted = lr.predict(x);
            int correct = 0;
            for (int i = 0; i < n; i++) {
                if (predicted[i] == y[i]) {
                    correct++;
                }
            }
            assertTrue("gradient descent must learn x0 > x1 (got " + correct + "/" + n + ")",
                    correct >= (int) (n * 0.95));
            float[] w = lr.getWeights();
            assertTrue("the rule's weights have opposite signs", w[0] > 0 && w[1] < 0);
        }
    }

    /**
     * Three iterations from zero parameters: the second and third depend on each sample's logit, so they check that
     * the training predictions use the features.
     */
    @Test
    public void iterationsMatchTheDoubleReference() {
        checkIterations(300, 7, 3, 71);
        checkIterations(19, 290, 3, 72);
    }

    /** A tolerance above every possible gradient stops training after the first iteration's step. */
    @Test
    public void convergenceStopsAfterTheFirstIterationBelowTolerance() {
        int n = 64;
        int f = 5;
        float[][] x = LinearReference.batch(81, n, f);
        int[] y = intLabels(82, n);
        float lr = 0.5f;
        try (LogisticRegression model = new LogisticRegression(backend, lr, 0.0f, 50, 2.0f)) {
            model.fit(x, y);
            reference(x, y, lr, 0.0f, 1).assertMatches("stopped fit", model.getWeights(), model.getBias());
        }
    }

    @Test
    public void zeroIterationsLeaveZeroParameters() {
        try (LogisticRegression model = new LogisticRegression(backend, 0.1f, 0.0f, 0, 1e-6f)) {
            model.fit(new float[][]{{1f, 2f}, {3f, 4f}}, new int[]{0, 1});
            TrainParity.assertBitIdentical("weights", new float[2], model.getWeights());
            assertEquals(0, Float.floatToRawIntBits(model.getBias()));
        }
    }

    @Test
    public void predictionsMatchTheFittedParameters() {
        int n = 300;
        int f = 9;
        float[][] x = LinearReference.batch(91, n, f);
        int[] y = intLabels(92, n);
        try (LogisticRegression model = new LogisticRegression(backend, 0.5f, 0.01f, 20, 0.0f)) {
            model.fit(x, y);
            LinearReference.Unit fitted = LinearReference.Unit.exact(model.getWeights(), model.getBias());
            float[][] test = LinearReference.batch(93, n, f);
            float[] got = model.predictProbability(test);
            int[] labels = model.predict(test);
            float[] flat = LinearReference.flatten(test);
            for (int i = 0; i < n; i++) {
                LinearReference.assertWithin("probability " + i, LinearReference.sigmoid(fitted.z(flat, i * f)),
                        got[i], LinearReference.sigmoidErr(fitted.zErr(flat, i * f)));
                assertEquals("label " + i, got[i] >= 0.5f ? 1 : 0, labels[i]);
            }
        }
    }

    /** Releasing the backend's train context after a fit (as {@code clearCache} does) leaves the model usable. */
    @Test
    public void predictsAfterTheTrainContextIsReleased() {
        float[][] x = LinearReference.batch(111, 40, 3);
        try (LogisticRegression model = new LogisticRegression(backend, 0.5f, 0.0f, 10, 0.0f)) {
            model.fit(x, intLabels(112, 40));
            float[] before = model.predictProbability(x);
            TrainGpu.release(backend);
            TrainParity.assertBitIdentical("probabilities", before, model.predictProbability(x));
            assertEquals(3, model.getWeights().length);
        }
    }

    @Test
    public void refitReplacesTheModel() {
        try (LogisticRegression model = new LogisticRegression(backend, 0.5f, 0.0f, 5, 0.0f)) {
            model.fit(LinearReference.batch(101, 10, 3), intLabels(102, 10));
            float[][] x = LinearReference.batch(103, 12, 4);
            int[] y = intLabels(104, 12);
            model.fit(x, y);
            assertEquals(4, model.getWeights().length);
            reference(x, y, 0.5f, 0.0f, 5).assertMatches("refit", model.getWeights(), model.getBias());
            assertArrayEquals(new float[0], model.predictProbability(new float[0][]), 0.0f);
        }
    }

    @Test(expected = IllegalStateException.class)
    public void rejectsPredictionBeforeFit() {
        try (LogisticRegression model = new LogisticRegression(backend)) {
            model.predict(new float[][]{{1f}});
        }
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsLabelsOtherThanZeroAndOne() {
        try (LogisticRegression model = new LogisticRegression(backend)) {
            model.fit(new float[][]{{1f}, {2f}}, new int[]{0, 2});
        }
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsRaggedSamples() {
        try (LogisticRegression model = new LogisticRegression(backend)) {
            model.fit(new float[][]{{1f, 2f}, {3f}}, new int[]{0, 1});
        }
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsEmptyTrainingData() {
        try (LogisticRegression model = new LogisticRegression(backend)) {
            model.fit(new float[0][], new int[0]);
        }
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsPredictionWithTheWrongFeatureCount() {
        try (LogisticRegression model = new LogisticRegression(backend, 0.1f, 0.0f, 1, 0.0f)) {
            model.fit(new float[][]{{1f, 2f}}, new int[]{1});
            model.predictProbability(new float[][]{{1f}});
        }
    }

    @Test(expected = IllegalStateException.class)
    public void rejectsUseAfterClose() {
        LogisticRegression model = new LogisticRegression(backend, 0.1f, 0.0f, 1, 0.0f);
        model.fit(new float[][]{{1f}}, new int[]{1});
        model.close();
        model.close();
        model.getWeights();
    }

    private void checkIterations(int n, int f, int iterations, long seed) {
        float[][] x = LinearReference.batch(seed, n, f);
        int[] y = intLabels(seed + 1, n);
        float lr = 0.4f;
        float reg = 0.02f;
        try (LogisticRegression model = new LogisticRegression(backend, lr, reg, iterations, 0.0f)) {
            model.fit(x, y);
            reference(x, y, lr, reg, iterations).assertMatches(iterations + " iterations (" + n + "x" + f + ")",
                    model.getWeights(), model.getBias());
        }
    }

    /** {@code iterations} full-batch steps from zero parameters in double, with the float paths' error budget. */
    private static LinearReference.Unit reference(float[][] x, int[] y, float lr, float reg, int iterations) {
        int n = x.length;
        int f = x[0].length;
        float[] flat = LinearReference.flatten(x);
        LinearReference.Unit unit = LinearReference.Unit.exact(new float[f], 0.0f);
        for (int it = 0; it < iterations; it++) {
            double[] e = new double[n];
            double[] eErr = new double[n];
            for (int i = 0; i < n; i++) {
                e[i] = LinearReference.sigmoid(unit.z(flat, i * f)) - y[i];
                eErr[i] = LinearReference.sigmoidErr(unit.zErr(flat, i * f)) + 1e-7;
            }
            double[][] g = LinearReference.gradient(flat, n, f, null, e, eErr);
            unit = unit.step(g[0], g[1], lr, reg);
        }
        return unit;
    }

    private static int[] intLabels(long seed, int n) {
        Random r = new Random(seed);
        int[] y = new int[n];
        for (int i = 0; i < n; i++) {
            y[i] = r.nextBoolean() ? 1 : 0;
        }
        return y;
    }
}

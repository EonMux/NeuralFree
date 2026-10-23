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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * {@link ForecastUnit} on the CPU reference path (a host backend without train kernels) and on every GPU API with
 * the train kernels available (skipped elsewhere): online learning of both heads, the Huber bound, parameter round
 * trips, input validation, and parity of the forward pass and of one SGD step with a double-precision reference
 * under the tolerance classes of the {@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}.
 */
@RunWith(Parameterized.class)
public class ForecastUnitTest {

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> backends() {
        return LinearReference.backends();
    }

    private final TrainParity.Api api;
    private TrainParity.Session session;
    private ComputeBackend backend;

    public ForecastUnitTest(String label, TrainParity.Api api) {
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
    public void directionHeadLearnsALinearRuleOnSoftLabels() {
        try (ForecastUnit u = new ForecastUnit(backend, 2, 0.5f, 0.0f)) {
            Random rng = new Random(42);
            for (int step = 0; step < 400; step++) {
                float[][] x = new float[16][2];
                float[] yd = new float[16];
                float[] ym = new float[16];
                for (int i = 0; i < 16; i++) {
                    x[i][0] = rng.nextFloat() * 2 - 1;
                    x[i][1] = rng.nextFloat() * 2 - 1;
                    boolean up = x[i][0] > x[i][1];
                    yd[i] = up ? 0.9f : 0.1f;
                    ym[i] = up ? 0.5f : -0.5f;
                }
                u.update(x, yd, ym);
            }
            assertEquals(400L * 16, u.samplesSeen());

            int correct = 0;
            int total = 500;
            for (int i = 0; i < total; i++) {
                float a = rng.nextFloat() * 2 - 1;
                float b = rng.nextFloat() * 2 - 1;
                boolean predicted = u.predictOne(new float[]{a, b})[0] >= 0.5f;
                if (predicted == (a > b)) {
                    correct++;
                }
            }
            assertTrue("online SGD must learn x0 > x1 (got " + correct + "/" + total + ")",
                    correct >= (int) (total * 0.95));
        }
    }

    /** ym = 0.5 x0 - 0.3 x1 stays inside the Huber bound, so the head converges to the least-squares solution. */
    @Test
    public void magnitudeHeadRegressesALinearTarget() {
        try (ForecastUnit u = new ForecastUnit(backend, 2, 0.2f, 0.0f)) {
            Random rng = new Random(7);
            for (int step = 0; step < 1500; step++) {
                float[][] x = new float[16][2];
                float[] yd = new float[16];
                float[] ym = new float[16];
                for (int i = 0; i < 16; i++) {
                    x[i][0] = rng.nextFloat() * 2 - 1;
                    x[i][1] = rng.nextFloat() * 2 - 1;
                    ym[i] = 0.5f * x[i][0] - 0.3f * x[i][1];
                    yd[i] = 0.5f;
                }
                u.update(x, yd, ym);
            }
            float[] w = u.moveWeights();
            assertEquals(0.5f, w[0], 0.05f);
            assertEquals(-0.3f, w[1], 0.05f);
            assertEquals(0.0f, u.moveBias(), 0.05f);
            float[] f = u.predictOne(new float[]{0.8f, -0.4f});
            assertEquals(0.5 * 0.8 - 0.3 * -0.4, f[1], 0.06);
        }
    }

    @Test
    public void parametersRoundTripAndDriveTheForwardPass() {
        try (ForecastUnit u = new ForecastUnit(backend, 3)) {
            float[] wd = {0.5f, -1.25f, 2.0f};
            float[] wm = {-0.75f, 0.25f, 1.5f};
            u.setParameters(wd, 0.75f, wm, -0.5f);
            TrainParity.assertBitIdentical("direction weights", wd, u.directionWeights());
            TrainParity.assertBitIdentical("move weights", wm, u.moveWeights());
            assertEquals(Float.floatToRawIntBits(0.75f), Float.floatToRawIntBits(u.directionBias()));
            assertEquals(Float.floatToRawIntBits(-0.5f), Float.floatToRawIntBits(u.moveBias()));

            float[] x = {1.0f, 2.0f, -0.5f};
            double zd = 0.75 + 0.5 * 1.0 + (-1.25) * 2.0 + 2.0 * (-0.5);
            double zm = -0.5 + (-0.75) * 1.0 + 0.25 * 2.0 + 1.5 * (-0.5);
            float[] f = u.predictOne(x);
            assertEquals(1.0 / (1.0 + Math.exp(-zd)), f[0], 1e-5);
            assertEquals(zm, f[1], 1e-5);
        }
    }

    /**
     * For one repeated feature vector graded 1 at weight 3 and 0 at weight 1, the weighted logistic MLE is the
     * weighted label mean 3/4, and the magnitude head's weighted least-squares estimate is the weighted move mean
     * (3 * 0.5 + 1 * (-0.5)) / 4 = 0.25. lr 0.1 keeps the magnitude head's linear dynamics contracting (with x = 1
     * the weight and bias receive identical gradients, and lr 0.5 is an exact period-2 cycle).
     */
    @Test
    public void sampleWeightsScaleEachSamplesInfluence() {
        float[][] x = {{1.0f}, {1.0f}};
        float[] yd = {1.0f, 0.0f};
        float[] ym = {0.5f, -0.5f};
        try (ForecastUnit weighted = new ForecastUnit(backend, 1, 0.1f, 0.0f);
             ForecastUnit unweighted = new ForecastUnit(backend, 1, 0.1f, 0.0f)) {
            for (int e = 0; e < 800; e++) {
                weighted.update(x, yd, ym, new float[]{3.0f, 1.0f});
                unweighted.update(x, yd, ym);
            }
            float[] fw = weighted.predictOne(new float[]{1.0f});
            float[] fu = unweighted.predictOne(new float[]{1.0f});
            assertEquals(0.75, fw[0], 0.02);
            assertEquals(0.25, fw[1], 0.02);
            assertEquals(0.5, fu[0], 0.02);
            assertEquals(0.0, fu[1], 0.02);
        }
    }

    /** One outlier move label of 1000 moves the 1-feature weight by at most lr x delta in one step. */
    @Test
    public void huberBoundsAnOutliersInfluenceOnTheMagnitudeHead() {
        try (ForecastUnit u = new ForecastUnit(backend, 1, 0.5f, 0.0f)) {
            u.update(new float[][]{{1.0f}}, new float[]{0.5f}, new float[]{1000.0f});
            float w = u.moveWeights()[0];
            assertTrue("one outlier step must be Huber-bounded (got " + w + ")", w <= 0.5f * 1.0f + 1e-6f);
            assertTrue("the step must still move toward the target", w > 0);
        }
    }

    @Test
    public void emptyBatchesNeedNoWork() {
        try (ForecastUnit u = new ForecastUnit(backend, 2)) {
            assertEquals(0, u.predict(new float[0][]).length);
            u.update(new float[0][], new float[0], new float[0]);
            assertEquals(0L, u.samplesSeen());
        }
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsSamplesWithTheWrongFeatureCount() {
        try (ForecastUnit u = new ForecastUnit(backend, 4)) {
            u.predictOne(new float[]{1, 2, 3});
        }
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsDirectionLabelsOutsideTheUnitInterval() {
        try (ForecastUnit u = new ForecastUnit(backend, 1)) {
            u.update(new float[][]{{1.0f}}, new float[]{1.5f}, new float[]{0.0f});
        }
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsNonFiniteMoveLabels() {
        try (ForecastUnit u = new ForecastUnit(backend, 1)) {
            u.update(new float[][]{{1.0f}}, new float[]{0.5f}, new float[]{Float.NaN});
        }
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsNegativeSampleWeights() {
        try (ForecastUnit u = new ForecastUnit(backend, 1)) {
            u.update(new float[][]{{1.0f}}, new float[]{1.0f}, new float[]{0.0f}, new float[]{-1.0f});
        }
    }

    /**
     * Releasing the backend's train context (as the dispatchers' {@code clearCache} does) leaves a unit usable: its
     * next step reopens the context and computes exactly what a unit created afterwards computes.
     */
    @Test
    public void keepsWorkingAfterTheTrainContextIsReleased() {
        float[] wd = {0.25f, -0.5f};
        float[] wm = {-0.75f, 0.5f};
        float[][] x = {{1.0f, 2.0f}, {-0.5f, 0.75f}};
        float[] yd = {0.9f, 0.2f};
        float[] ym = {2.5f, -0.25f};
        try (ForecastUnit before = new ForecastUnit(backend, 2, 0.5f, 0.01f)) {
            before.setParameters(wd, 0.125f, wm, -0.25f);
            TrainGpu.release(backend);
            before.update(x, yd, ym);
            try (ForecastUnit after = new ForecastUnit(backend, 2, 0.5f, 0.01f)) {
                after.setParameters(wd, 0.125f, wm, -0.25f);
                after.update(x, yd, ym);
                TrainParity.assertBitIdentical("direction weights", after.directionWeights(),
                        before.directionWeights());
                TrainParity.assertBitIdentical("move weights", after.moveWeights(), before.moveWeights());
                assertEquals(Float.floatToRawIntBits(after.directionBias()),
                        Float.floatToRawIntBits(before.directionBias()));
                assertEquals(Float.floatToRawIntBits(after.moveBias()), Float.floatToRawIntBits(before.moveBias()));
            }
        }
    }

    /** Many samples (several forward work-groups) and many features (several gradient work-groups). */
    @Test
    public void forwardMatchesTheDoubleReference() {
        checkForward(300, 7, 41);
        checkForward(19, 290, 42);
    }

    @Test
    public void oneWeightedStepMatchesTheDoubleReference() {
        checkStep(300, 7, 51, true);
        checkStep(19, 290, 52, true);
    }

    @Test
    public void oneUnweightedStepMatchesTheDoubleReference() {
        checkStep(300, 7, 61, false);
        checkStep(19, 290, 62, false);
    }

    private void checkForward(int n, int f, long seed) {
        float[] wd = TrainParity.uniform(seed, f, -0.5f, 0.5f);
        float[] wm = TrainParity.uniform(seed + 1, f, -0.5f, 0.5f);
        float[] biases = TrainParity.uniform(seed + 2, 2, -0.5f, 0.5f);
        float[][] x = LinearReference.batch(seed + 3, n, f);
        try (ForecastUnit u = new ForecastUnit(backend, f)) {
            u.setParameters(wd, biases[0], wm, biases[1]);
            float[][] got = u.predict(x);
            LinearReference.Unit dir = LinearReference.Unit.exact(wd, biases[0]);
            LinearReference.Unit move = LinearReference.Unit.exact(wm, biases[1]);
            float[] flat = LinearReference.flatten(x);
            for (int i = 0; i < n; i++) {
                int row = i * f;
                LinearReference.assertWithin("direction " + i + " (" + n + "x" + f + ")",
                        LinearReference.sigmoid(dir.z(flat, row)), got[i][0],
                        LinearReference.sigmoidErr(dir.zErr(flat, row)));
                LinearReference.assertWithin("move " + i + " (" + n + "x" + f + ")",
                        move.z(flat, row), got[i][1], move.zErr(flat, row));
            }
        }
    }

    /** Move labels in [-3, 3] put part of the batch beyond the Huber bound. */
    private void checkStep(int n, int f, long seed, boolean weighted) {
        float lr = 0.3f;
        float l2 = 0.05f;
        float[] wd = TrainParity.uniform(seed, f, -0.5f, 0.5f);
        float[] wm = TrainParity.uniform(seed + 1, f, -0.5f, 0.5f);
        float[] biases = TrainParity.uniform(seed + 2, 2, -0.5f, 0.5f);
        float[][] x = LinearReference.batch(seed + 3, n, f);
        float[] yd = TrainParity.uniform(seed + 4, n, 0f, 1f);
        float[] ym = TrainParity.uniform(seed + 5, n, -3f, 3f);
        float[] s = weighted ? TrainParity.uniform(seed + 6, n, 0f, 3f) : null;
        try (ForecastUnit u = new ForecastUnit(backend, f, lr, l2)) {
            u.setParameters(wd, biases[0], wm, biases[1]);
            u.update(x, yd, ym, s);

            LinearReference.Unit dir = LinearReference.Unit.exact(wd, biases[0]);
            LinearReference.Unit move = LinearReference.Unit.exact(wm, biases[1]);
            float[] flat = LinearReference.flatten(x);
            double[] ed = new double[n];
            double[] edErr = new double[n];
            double[] em = new double[n];
            double[] emErr = new double[n];
            for (int i = 0; i < n; i++) {
                int row = i * f;
                ed[i] = LinearReference.sigmoid(dir.z(flat, row)) - yd[i];
                edErr[i] = LinearReference.sigmoidErr(dir.zErr(flat, row)) + 1e-7;
                double v = move.z(flat, row);
                em[i] = Math.max(-ForecastUnit.HUBER_DELTA, Math.min(ForecastUnit.HUBER_DELTA, v - ym[i]));
                emErr[i] = move.zErr(flat, row) + LinearReference.ROUNDING * (Math.abs(v) + Math.abs(ym[i]));
            }
            double[][] gd = LinearReference.gradient(flat, n, f, s, ed, edErr);
            double[][] gm = LinearReference.gradient(flat, n, f, s, em, emErr);
            String what = "step (" + n + "x" + f + ")";
            dir.step(gd[0], gd[1], lr, l2).assertMatches(what + " direction head", u.directionWeights(),
                    u.directionBias());
            move.step(gm[0], gm[1], lr, l2).assertMatches(what + " magnitude head", u.moveWeights(), u.moveBias());
        }
    }
}

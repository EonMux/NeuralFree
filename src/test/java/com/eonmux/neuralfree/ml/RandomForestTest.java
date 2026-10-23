package com.eonmux.neuralfree.ml;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * {@link RandomForest}: with the default feature-subset rule, regression on one and two features must split on a
 * feature (the {@code nFeatures / 3} default is floored at one, and a split keeps examining features until one yields
 * a valid partition), classification on one feature separates classes, and hyperparameters and input shapes are
 * validated.
 */
public class RandomForestTest {

    private ComputeBackend host;

    @Before
    public void setUp() {
        host = TrainParity.hostBackend();
    }

    /** {@code n} evenly spaced points in [0, 1) in column {@code signal} of {@code nFeatures} columns; others are 0. */
    private static float[][] grid(int n, int nFeatures, int signal) {
        float[][] x = new float[n][nFeatures];
        for (int i = 0; i < n; i++) {
            x[i][signal] = i / (float) n;
        }
        return x;
    }

    @Test
    public void regressionOnOneFeatureFollowsTheInput() {
        float[][] x = grid(40, 1, 0);
        float[] y = new float[40];
        for (int i = 0; i < 40; i++) {
            y[i] = 2.0f * x[i][0];
        }
        float[] pred;
        try (RandomForest rf = new RandomForest(host, 10)) {
            rf.fitRegression(x, y);
            pred = rf.predictRegression(new float[][]{{0.1f}, {0.5f}, {0.9f}});
        }
        assertTrue("predictions increase with x: " + pred[0] + ", " + pred[1] + ", " + pred[2],
                pred[0] < pred[1] && pred[1] < pred[2]);
        assertEquals(0.2f, pred[0], 0.15f);
        assertEquals(1.8f, pred[2], 0.15f);
    }

    @Test
    public void regressionOnTwoFeaturesSplitsOnTheSignalFeature() {
        float[][] x = grid(40, 2, 1);
        float[] y = new float[40];
        for (int i = 0; i < 40; i++) {
            y[i] = x[i][1] < 0.5f ? -1.0f : 1.0f;
        }
        float[] pred;
        try (RandomForest rf = new RandomForest(host, 10)) {
            rf.fitRegression(x, y);
            pred = rf.predictRegression(new float[][]{{0.0f, 0.2f}, {0.0f, 0.8f}});
        }
        assertEquals(-1.0f, pred[0], 0.2f);
        assertEquals(1.0f, pred[1], 0.2f);
    }

    @Test
    public void classificationOnOneFeatureSeparatesClasses() {
        float[][] x = grid(40, 1, 0);
        int[] y = new int[40];
        for (int i = 0; i < 40; i++) {
            y[i] = x[i][0] < 0.5f ? 0 : 1;
        }
        int[] pred;
        try (RandomForest rf = new RandomForest(host, 10)) {
            rf.fitClassification(x, y);
            pred = rf.predictClassification(new float[][]{{0.1f}, {0.9f}});
        }
        assertArrayEquals(new int[]{0, 1}, pred);
    }

    private static void assertRejected(Runnable call, String messagePart) {
        try {
            call.run();
            fail("expected IllegalArgumentException mentioning " + messagePart);
        } catch (IllegalArgumentException e) {
            assertTrue("message '" + e.getMessage() + "' mentions " + messagePart,
                    e.getMessage().contains(messagePart));
        }
    }

    @Test
    public void constructorRejectsInvalidTreeHyperparameters() {
        assertRejected(() -> new RandomForest(host, 5, 0, 2, 1, -1, true, 1), "maxDepth");
        assertRejected(() -> new RandomForest(host, 5, 10, 1, 1, -1, true, 1), "minSamplesSplit");
        assertRejected(() -> new RandomForest(host, 5, 10, 2, 0, -1, true, 1), "minSamplesLeaf");
    }

    @Test
    public void fitAndPredictRejectMalformedInputs() {
        try (RandomForest rf = new RandomForest(host, 3)) {
            assertRejected(() -> rf.fitRegression(new float[0][], new float[0]), "at least one row");
            assertRejected(() -> rf.fitRegression(new float[][]{{1f}, {2f}}, new float[]{1f}), "rows");
            assertRejected(() -> rf.fitClassification(new float[][]{{1f, 2f}, {3f}}, new int[]{0, 1}), "x[1]");
            rf.fitRegression(new float[][]{{0f, 1f}, {1f, 0f}, {2f, 2f}}, new float[]{0f, 1f, 2f});
            assertRejected(() -> rf.predictRegression(new float[][]{{0f}}), "x[0]");
            assertEquals(0, rf.predictRegression(new float[0][]).length);
        }
    }
}

package com.eonmux.neuralfree.ml;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * {@link HiddenMarkovModel} input validation: constructor hyperparameters, and null sequences or symbols outside
 * {@code [0, nObservations)} in {@code fit}, {@code predict} and {@code score}, reported with their position.
 */
public class HiddenMarkovModelTest {

    private static final int[][] TRAINING = {{0, 1, 2, 1, 0}, {2, 2, 1, 0}};

    private ComputeBackend host;

    @Before
    public void setUp() {
        host = TrainParity.hostBackend();
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
    public void constructorRejectsInvalidHyperparameters() {
        assertRejected(() -> new HiddenMarkovModel(host, 2, 3, -1, 1e-6f), "maxIterations");
        assertRejected(() -> new HiddenMarkovModel(host, 2, 3, 10, -1e-6f), "tolerance");
        assertRejected(() -> new HiddenMarkovModel(host, 2, 3, 10, Float.NaN), "tolerance");
    }

    @Test
    public void fitRejectsNullAndOutOfRangeSequences() {
        try (HiddenMarkovModel hmm = new HiddenMarkovModel(host, 2, 3)) {
            assertRejected(() -> hmm.fit(new int[][]{{0, 1}, null}), "sequences[1]");
            assertRejected(() -> hmm.fit(new int[][]{{0, 1}, {1, 2, 3}}), "sequences[1][2]");
            assertRejected(() -> hmm.fit(new int[][]{{-1, 0}}), "sequences[0][0]");
        }
    }

    @Test
    public void predictAndScoreRejectOutOfRangeSymbols() {
        try (HiddenMarkovModel hmm = new HiddenMarkovModel(host, 2, 3)) {
            hmm.fit(TRAINING);
            assertRejected(() -> hmm.predict(new int[]{0, 3}), "sequence[1]");
            assertRejected(() -> hmm.score(new int[]{-2}), "sequence[0]");
        }
    }

    @Test
    public void validSequencesFitAndDecode() {
        try (HiddenMarkovModel hmm = new HiddenMarkovModel(host, 2, 3)) {
            hmm.fit(TRAINING);
            int[] path = hmm.predict(new int[]{0, 1, 2});
            assertEquals(3, path.length);
            for (int state : path) {
                assertTrue(state == 0 || state == 1);
            }
            float logLikelihood = hmm.score(new int[]{0, 1, 2});
            assertTrue("log-likelihood is finite and non-positive: " + logLikelihood,
                    Float.isFinite(logLikelihood) && logLikelihood <= 0.0f);
        }
    }
}

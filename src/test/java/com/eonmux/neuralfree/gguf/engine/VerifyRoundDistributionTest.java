package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.engine.draft.RejectionVerifier;
import com.eonmux.neuralfree.gguf.engine.draft.Verifier;
import org.junit.Test;

import java.util.SplittableRandom;
import java.util.function.IntFunction;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Rejection sampling through a {@link VerifyRound} (the round the batch engine and {@link SpeculativeDecoder} run),
 * sampled: proposals drawn from a draft distribution {@code q} and verified against rows whose chain distribution
 * is {@code p}, with {@code q} handed in as the drafter's proposal rows, emit exactly the target law at every
 * position (Pearson goodness of fit), and a drafter misreporting {@code q} measurably does not; a deterministic
 * proposal reported as one-hot stays lossless.
 */
public class VerifyRoundDistributionTest {

    private static final int TRIALS = 20_000;
    private static final double[] P0 = {0.4, 0.3, 0.2, 0.1};
    private static final double[] P1 = {0.05, 0.25, 0.5, 0.2};
    private static final double[] Q0 = {0.1, 0.2, 0.3, 0.4};
    private static final double[] Q1 = {0.4, 0.3, 0.2, 0.1};
    /** The upper 0.0005 quantile of the chi-square law with 3 degrees of freedom. */
    private static final double CRITICAL_3 = 17.73;

    /** The rows of a two-token round: the logits whose softmax is {@code p} at each position, the bonus last. */
    private static float[][] rows() {
        return new float[][]{logits(P0), logits(P1), logits(new double[]{0.25, 0.25, 0.25, 0.25})};
    }

    private static float[] logits(double[] p) {
        float[] l = new float[p.length];
        for (int i = 0; i < p.length; i++) {
            l[i] = (float) Math.log(p[i]);
        }
        return l;
    }

    private static float[] row(double[] q) {
        float[] r = new float[q.length];
        for (int i = 0; i < q.length; i++) {
            r[i] = (float) q[i];
        }
        return r;
    }

    private static int draw(double[] q, SplittableRandom rng) {
        double u = rng.nextDouble();
        double run = 0;
        for (int i = 0; i < q.length; i++) {
            run += q[i];
            if (u < run) {
                return i;
            }
        }
        return q.length - 1;
    }

    /** The emitted tokens of one round, in order. */
    private static final class Recorder implements VerifyRound.Emitter {
        final int[] tokens = new int[3];
        int count;

        @Override public boolean cancelled() {
            return false;
        }

        @Override public StopReason offer(int token, float[] row, Verifier.Distribution processed) {
            tokens[count++] = token;
            return null;
        }
    }

    /** Counts of the first emission and of the second (after a confirmed first draft token) over the trials. */
    private static int[][] run(IntFunction<float[]> reported, boolean drawn) {
        SamplerConfig chain = SamplerConfig.builder().temperature(1f).topK(0).topP(1f).minP(0f).seed(5).build();
        SplittableRandom drafts = new SplittableRandom(17);
        SplittableRandom uniforms = new SplittableRandom(29);
        int[][] counts = new int[2][4];
        for (int t = 0; t < TRIALS; t++) {
            int[] draft = drawn ? new int[]{draw(Q0, drafts), draw(Q1, drafts)} : new int[]{0, 2};
            Recorder out = new Recorder();
            RejectionVerifier verifier = new RejectionVerifier();
            VerifyRound round = new VerifyRound(Sampler.create(chain), out, draft, rows(), 0, uniforms, reported);
            verifier.verify(round);
            round.finish(verifier);
            assertTrue(out.count >= 1 && round.emitted() == out.count);
            counts[0][out.tokens[0]]++;
            if (round.accepted() >= 1) {
                counts[1][out.tokens[1]]++;
            }
        }
        return counts;
    }

    private static double chiSquare(int[] counts, double[] p) {
        int n = 0;
        for (int c : counts) {
            n += c;
        }
        double x2 = 0;
        for (int i = 0; i < p.length; i++) {
            double expected = n * p[i];
            x2 += (counts[i] - expected) * (counts[i] - expected) / expected;
        }
        return x2;
    }

    @Test
    public void drawnProposalsWithTheirDistributionsEmitTheTargetLawAtEveryPosition() {
        int[][] counts = run(position -> row(position == 0 ? Q0 : Q1), true);
        double first = chiSquare(counts[0], P0);
        double second = chiSquare(counts[1], P1);
        assertTrue("first emission chi2 " + first, first < CRITICAL_3);
        assertTrue("second emission chi2 " + second, second < CRITICAL_3);
        // the confirmed-first measure is the acceptance-ratio sum, sum_d q(d) min(1, p(d) / q(d))
        double accept = 0;
        for (int d = 0; d < 4; d++) {
            accept += Math.min(Q0[d], P0[d]);
        }
        int confirmed = 0;
        for (int c : counts[1]) {
            confirmed += c;
        }
        assertEquals(accept, confirmed / (double) TRIALS, 0.015);
    }

    @Test
    public void aMisreportedProposalDistributionIsDetected() {
        double[] uniform = {0.25, 0.25, 0.25, 0.25};
        int[][] counts = run(position -> row(uniform), true);
        // q drawn but uniform reported: the first emission law is (0.325, 0.275, 0.24, 0.16), not P0
        double first = chiSquare(counts[0], P0);
        assertTrue("first emission chi2 " + first, first > 10 * CRITICAL_3);
        assertTrue(chiSquare(counts[0], new double[]{0.325, 0.275, 0.24, 0.16}) < CRITICAL_3);
    }

    @Test
    public void aFixedProposalReportedOneHotStaysLossless() {
        int[][] counts = run(position -> null, false);
        double first = chiSquare(counts[0], P0);
        double second = chiSquare(counts[1], P1);
        assertTrue("first emission chi2 " + first, first < CRITICAL_3);
        assertTrue("second emission chi2 " + second, second < CRITICAL_3);
    }
}

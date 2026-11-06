package com.eonmux.neuralfree.gguf.engine.draft;

import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.SpeculativeDecoder;
import com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.FakeTarget;
import com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.PlainDrafter;
import com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.Run;
import com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.ScriptRound;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.logits;
import static com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.request;
import static com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.sampling;
import static com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.tokenizer;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.VOCAB;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.array;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.assertEq;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.cdfBreakpoints;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.enumerate;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.expectedLength;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.firstEmissionLaw;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.measure;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.measureConfirmed;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.minRatios;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.plusInto;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.residualWeights;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.round;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.sum;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.thresholds;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.twoTokenRound;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.union;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.weighted;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Block verification ({@link BlockVerifier}, Sun et al., Algorithm 3) on a 4-token vocabulary: distribution
 * preservation and the block coupling's confirmed-length and residual laws by exact enumeration, an accepted length
 * never below token-level rejection's, unconditional acceptance of a block whose joint ratio reaches one, and the
 * block's prefix and residual through the SpeculativeDecoder.
 */
public class BlockVerifierTest {

    /**
     * Block verification (Sun et al., Algorithm 3): the same lossless law under the block coupling's
     * confirmed-length measures, and an accepted length never below token-level rejection's for the same inputs.
     */
    @Test
    public void blockVerificationPreservesTheTargetDistributionByExactEnumeration() {
        double[] p0 = array(0.4, 0.3, 0.2, 0.1);
        double[] q0 = array(0.1, 0.2, 0.3, 0.4);
        double[] p1 = array(0.05, 0.25, 0.5, 0.2);
        double[] q1 = array(0.4, 0.3, 0.2, 0.1);
        double[] a0 = minRatios(p0, q0);
        double[] a1 = minRatios(p1, q1);
        double[] w0 = residualWeights(p0, q0);
        double[] w1 = residualWeights(p1, q1);

        double[] depth0 = new double[VOCAB * VOCAB];
        double[] depth1 = new double[VOCAB * VOCAB];
        double[][] a0Of = new double[VOCAB][VOCAB];
        double[][] a1Of = new double[VOCAB][VOCAB];
        double[][][] w0Of = new double[VOCAB][VOCAB][];
        double[][][] w1Of = new double[VOCAB][VOCAB][];
        int n = 0;
        for (int d0 = 0; d0 < VOCAB; d0++) {
            for (int d1 = 0; d1 < VOCAB; d1++) {
                double mb = p0[d0] * p1[d1];
                double ms = q0[d0] * q1[d1];
                a0Of[d0][d1] = mb < ms ? mb / ms : 1.0;
                double bm1 = p0[d0];
                double ms1 = q0[d0];
                double remain1 = weightedPositive(p1, q1, bm1, ms1, true);
                double rej1 = weightedPositive(p1, q1, bm1, ms1, false);
                a1Of[d0][d1] = rej1 > 0 ? (remain1 >= rej1 ? 1.0 : remain1 / rej1) : 1.0;
                w0Of[d0][d1] = normalizedPositive(p0, q0, 1.0, 1.0);
                w1Of[d0][d1] = normalizedPositive(p1, q1, bm1, ms1);
                depth0[n] = a0Of[d0][d1];
                depth1[n] = a1Of[d0][d1];
                n++;
            }
        }
        double[] cdfs0 = cdfsOf(w0Of);
        double[] cdfs1 = cdfsOf(w1Of);
        List<Run> total = new ArrayList<>();
        for (int d0 = 0; d0 < VOCAB; d0++) {
            for (int d1 = 0; d1 < VOCAB; d1++) {
                List<Run> runs = enumerate(new BlockVerifier(),
                        twoTokenRound(p0, p1, q0, q1, d0, d1),
                        thresholds(depth0), thresholds(depth1), thresholds(union(cdfs0, cdfs1)));
                plusInto(total, weighted(runs, q0[d0] * q1[d1]));
            }
        }
        assertEq(1.0, sum(firstEmissionLaw(total)), 1e-9, "block total measure");
        double[] law = firstEmissionLaw(total);
        for (int x = 0; x < VOCAB; x++) {
            assertEq(p0[x], law[x], 1e-9, "block emission law for token " + x);
        }
        double m2 = 0.0;
        double m1 = 0.0;
        for (int d0 = 0; d0 < VOCAB; d0++) {
            for (int d1 = 0; d1 < VOCAB; d1++) {
                m2 += q0[d0] * q1[d1] * a0Of[d0][d1];
                m1 += q0[d0] * q1[d1] * (1.0 - a0Of[d0][d1]) * a1Of[d0][d1];
            }
        }
        double m0 = 1.0 - m2 - m1;
        assertEq(m2, measureConfirmed(total, 2), 1e-9, "block confirmed-2 measure");
        assertEq(m1, measureConfirmed(total, 1), 1e-9, "block confirmed-1 measure");
        assertEq(m0, measureConfirmed(total, 0), 1e-9, "block confirmed-0 measure");
        for (int x = 0; x < VOCAB; x++) {
            assertEq(w0Of[0][0][x], measure(total, 0, x) / measureConfirmed(total, 0), 1e-9,
                    "block residual law after 0 confirms, token " + x);
            // after 1 confirm the residual weights scale with the confirmed proposal's masses,
            // so the law is the proposal-weighted mixture of the per-proposal residuals
            double mix1 = 0.0;
            for (int d0 = 0; d0 < VOCAB; d0++) {
                for (int d1 = 0; d1 < VOCAB; d1++) {
                    mix1 += q0[d0] * q1[d1] * (1.0 - a0Of[d0][d1]) * a1Of[d0][d1] * w1Of[d0][d1][x];
                }
            }
            assertEq(mix1 / m1, measure(total, 1, x) / measureConfirmed(total, 1), 1e-9,
                    "block residual law after 1 confirm, token " + x);
        }
        // the bonus after a fully confirmed block is the chain draw of the last row (scripted: token 3).
        assertEq(m2, measure(total, 2, 3), 1e-9, "block bonus law");
        // and block verification is never worse in expected accepted length than token-level rejection.
        List<Run> rejectionRuns = new ArrayList<>();
        for (int d0 = 0; d0 < VOCAB; d0++) {
            for (int d1 = 0; d1 < VOCAB; d1++) {
                List<Run> runs = enumerate(new RejectionVerifier(),
                        twoTokenRound(p0, p1, q0, q1, d0, d1),
                        thresholds(a0), thresholds(union(a1, cdfBreakpoints(w0))),
                        thresholds(cdfBreakpoints(w1)));
                plusInto(rejectionRuns, weighted(runs, q0[d0] * q1[d1]));
            }
        }
        assertTrue("block must not lose accepted length: " + expectedLength(total) + " vs "
                        + expectedLength(rejectionRuns),
                expectedLength(total) >= expectedLength(rejectionRuns) - 1e-12);
    }

    private static double[] cdfsOf(double[][][] weights) {
        List<Double> out = new ArrayList<>();
        for (double[][] entry : weights) {
            for (double[] w : entry) {
                for (double t : cdfBreakpoints(w)) {
                    out.add(t);
                }
            }
        }
        double[] a = new double[out.size()];
        for (int i = 0; i < a.length; i++) {
            a[i] = out.get(i);
        }
        return a;
    }

    /** {@code sum_x (a·p(x) − b·q(x))+} (remain) or the negative-side sum (rej). */
    private static double weightedPositive(double[] p, double[] q, double a, double b, boolean remain) {
        double total = 0.0;
        for (int i = 0; i < p.length; i++) {
            double w = remain ? a * p[i] - b * q[i] : b * q[i] - a * p[i];
            if (w > 0.0) {
                total += w;
            }
        }
        return total;
    }

    /** {@code (a·p − b·q)+}, normalized. */
    private static double[] normalizedPositive(double[] p, double[] q, double a, double b) {
        double[] w = new double[p.length];
        double total = 0.0;
        for (int i = 0; i < p.length; i++) {
            w[i] = Math.max(0.0, a * p[i] - b * q[i]);
            total += w[i];
        }
        if (total > 0.0) {
            for (int i = 0; i < w.length; i++) {
                w[i] /= total;
            }
        }
        return w;
    }

    /**
     * A whole block whose joint ratio reaches one accepts unconditionally: the confirmed length is the full draft,
     * and only the opening uniform is consumed.
     */
    @Test
    public void blockVerificationAcceptsAWholeBlockWhoseJointRatioIsAtLeastOne() {
        double[] p = array(0.4, 0.3, 0.2, 0.1);
        double[] q = array(0.1, 0.2, 0.33, 0.37);
        // proposal (0, 1): Mb = 0.4·0.3 = 0.12, Ms = 0.1·0.2 = 0.02, ratio 6.
        ScriptRound only = round(new BlockVerifier(), twoTokenRound(p, p, q, q, 0, 1), new double[]{0.5});
        assertEquals(2, only.confirmed);
        assertEquals(Arrays.asList(0, 1, 3), only.emitted);
        assertEquals(1, only.uniformsUsed);
    }

    // ---------------------------------------------------------------- through the decoder

    /** Block verification on one-hot rows: the block's backward pass recovers the argmax prefix and replacement. */
    @Test
    public void blockVerificationConfirmsThePrefixAndRescoresFromItsResidual() {
        FakeTarget target = new FakeTarget(false)
                .add(logits(0.0, 1.0, 0.0, 0.0))
                // the third row is never read: the block's residual at position 1 ends the round
                .add(logits(0.0, 0.0, 1.0, 0.0), logits(1.0, 0.0, 0.0, 0.0), logits(0.0, 1.0, 0.0, 0.0))
                .add(logits(1.0, 0.0, 0.0, 0.0));
        List<SpeculativeDecoder.Stats> stats = new ArrayList<>();
        SpeculativeDecoder dec = SpeculativeDecoder.builder(target, new PlainDrafter(new int[]{2, 3}), tokenizer())
                .draftMax(2).verifyMode(VerifyMode.BLOCK).statsSink(stats::add).build();

        GenerationResult r = dec.generate(new int[]{0}, request(4, sampling()), (TokenCallback) null);
        assertArrayEquals(new int[]{1, 2, 0, 0}, r.tokens());
        assertEquals(1, stats.size());
        assertEquals(2, stats.get(0).drafted());
        assertArrayEquals(new int[]{1, 2, 3}, target.verifyCalls.get(0));
        assertEquals(Arrays.asList(1), target.rollbacks);
    }
}

package com.eonmux.neuralfree.gguf.engine.draft;

import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.SpeculativeDecoder;
import com.eonmux.neuralfree.gguf.engine.StopReason;
import com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.FakeTarget;
import com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.PlainDrafter;
import com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.RoundMaker;
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
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.dist;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.distSelected;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.enumerate;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.firstEmissionLaw;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.measure;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.measureConfirmed;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.minRatios;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.plusInto;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.residualWeights;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.sum;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.thresholds;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.twoTokenRound;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.union;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.weighted;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Token-level rejection sampling ({@link RejectionVerifier}) on a 4-token vocabulary: distribution preservation by
 * exact enumeration (every uniform cell of the confirmation regions, under every proposal the drafter's distribution
 * gives), the confirmed-length and residual laws, and a deterministic confirm-and-rescore trace through the
 * SpeculativeDecoder on KV memory.
 */
public class RejectionVerifierTest {

    /** Returns a k = 1 round on the 4-token vocabulary for the given proposal. */
    private static RoundMaker rejectionRound1(final double[] p0, final int bonusSel, final double[] q0,
                                              final int proposal) {
        return new RoundMaker() {
            @Override public ScriptRound round(double[] uniforms) {
                return new ScriptRound(
                        new Verifier.Distribution[]{dist(p0), distSelected(bonusSel, 1.0, 0, 0, 0)},
                        new Verifier.Distribution[]{dist(q0)}, new int[]{proposal}, uniforms);
            }
        };
    }

    /**
     * Token-level rejection sampling on a 4-token vocabulary with a probabilistic drafter: the enumerated emission
     * law is exactly the target's, and the confirmed-length law is the acceptance-ratio sum.
     */
    @Test
    public void rejectionSamplingPreservesTheTargetDistributionByExactEnumeration() {
        double[] p0 = array(0.4, 0.3, 0.2, 0.1);
        double[] q0 = array(0.1, 0.2, 0.3, 0.4);
        double[] w0 = residualWeights(p0, q0);
        double[] aOf = minRatios(p0, q0);

        List<Run> total = new ArrayList<>();
        for (int d = 0; d < VOCAB; d++) {
            List<Run> runs = enumerate(new RejectionVerifier(), rejectionRound1(p0, 2, q0, d),
                    thresholds(aOf), thresholds(cdfBreakpoints(w0)));
            plusInto(total, weighted(runs, q0[d]));
        }
        assertEq(1.0, sum(firstEmissionLaw(total)), 1e-9, "total emission measure");
        double[] law = firstEmissionLaw(total);
        for (int x = 0; x < VOCAB; x++) {
            assertEq(p0[x], law[x], 1e-9, "rejection emission law for token " + x);
        }
        double accepted = 0.0;
        for (int d = 0; d < VOCAB; d++) {
            accepted += q0[d] * aOf[d];
        }
        assertEq(accepted, measureConfirmed(total, 1), 1e-9, "rejection confirmed measure");
        assertEq(1.0 - accepted, measureConfirmed(total, 0), 1e-9, "rejection rejected measure");
        // the replacement after the rejection is the residual, exactly.
        double[] normalizedW0 = normalized(w0);
        for (int x = 0; x < VOCAB; x++) {
            if (normalizedW0[x] > 0.0) {
                assertEq(normalizedW0[x], measure(total, 0, x) / measureConfirmed(total, 0), 1e-9,
                        "rejection residual law for token " + x);
            }
        }
    }

    private static double[] normalized(double[] w) {
        double total = sum(w);
        double[] out = new double[w.length];
        for (int i = 0; i < w.length; i++) {
            out[i] = total > 0 ? w[i] / total : 0.0;
        }
        return out;
    }

    /**
     * Two-draft-token rejection sampling: the same lossless law over both positions, the confirmed-length law as
     * the product of the ratios, and both residual laws matching the target distributions.
     */
    @Test
    public void rejectionSamplingKeepsEveryPositionAlignedWithTheTargetByExactEnumeration() {
        double[] p0 = array(0.4, 0.3, 0.2, 0.1);
        double[] q0 = array(0.1, 0.2, 0.3, 0.4);
        double[] p1 = array(0.05, 0.25, 0.5, 0.2);
        double[] q1 = array(0.4, 0.3, 0.2, 0.1);
        double[] w0 = residualWeights(p0, q0);
        double[] w1 = residualWeights(p1, q1);
        double[] a0 = minRatios(p0, q0);
        double[] a1 = minRatios(p1, q1);

        double[] depth0 = thresholds(a0);
        double[] depth1 = thresholds(union(a1, cdfBreakpoints(w0)));
        double[] depth2 = thresholds(cdfBreakpoints(w1));
        List<Run> total = new ArrayList<>();
        for (int d0 = 0; d0 < VOCAB; d0++) {
            for (int d1 = 0; d1 < VOCAB; d1++) {
                List<Run> runs = enumerate(new RejectionVerifier(),
                        twoTokenRound(p0, p1, q0, q1, d0, d1), depth0, depth1, depth2);
                plusInto(total, weighted(runs, q0[d0] * q1[d1]));
            }
        }
        assertEq(1.0, sum(firstEmissionLaw(total)), 0.0, "total emission measure");
        double[] law = firstEmissionLaw(total);
        for (int x = 0; x < VOCAB; x++) {
            assertEq(p0[x], law[x], 1e-9, "two-position emission law for token " + x);
        }
        double m2 = 0.0;
        double m1 = 0.0;
        double m0 = 0.0;
        for (int d0 = 0; d0 < VOCAB; d0++) {
            for (int d1 = 0; d1 < VOCAB; d1++) {
                m2 += q0[d0] * q1[d1] * a0[d0] * a1[d1];
                m1 += q0[d0] * q1[d1] * a0[d0] * (1.0 - a1[d1]);
                m0 += q0[d0] * q1[d1] * (1.0 - a0[d0]);
            }
        }
        assertEq(m2, measureConfirmed(total, 2), 1e-9, "confirmed-2 measure");
        assertEq(m1, measureConfirmed(total, 1), 1e-9, "confirmed-1 measure");
        assertEq(m0, measureConfirmed(total, 0), 1e-9, "confirmed-0 measure");
        double[] nw0 = normalized(w0);
        double[] nw1 = normalized(w1);
        for (int x = 0; x < VOCAB; x++) {
            assertEq(nw0[x], measure(total, 0, x) / measureConfirmed(total, 0), 1e-9,
                    "replacement law after 0 confirms, token " + x);
            assertEq(nw1[x], measure(total, 1, x) / measureConfirmed(total, 1), 1e-9,
                    "replacement law after 1 confirm, token " + x);
        }
        // the bonus after a fully confirmed block is the chain draw of the last row (scripted: token 3).
        assertEq(m2, measure(total, 2, 3), 1e-9, "bonus law after 2 confirms");
    }

    // ---------------------------------------------------------------- through the decoder

    /** One-hot target rows and a plain drafter run rejection sampling deterministically on KV memory. */
    @Test
    public void rejectionSamplingConfirmsAndRescoresOnKvMemory() {
        FakeTarget target = new FakeTarget(false)
                .add(logits(0.0, 1.0, 0.0, 0.0))
                // the verify scores the whole [1, 2, 3] batch: row 2 is never read because the
                // rejection at position 1 ends the round before it
                .add(logits(0.0, 0.0, 1.0, 0.0), logits(1.0, 0.0, 0.0, 0.0), logits(0.0, 1.0, 0.0, 0.0))
                .add(logits(1.0, 0.0, 0.0, 0.0));
        List<SpeculativeDecoder.Stats> stats = new ArrayList<>();
        SpeculativeDecoder dec = SpeculativeDecoder.builder(target, new PlainDrafter(new int[]{2, 3}), tokenizer())
                .draftMax(2).verifyMode(VerifyMode.REJECTION).statsSink(stats::add).build();

        GenerationResult r = dec.generate(new int[]{0}, request(4, sampling()), (TokenCallback) null);
        assertArrayEquals(new int[]{1, 2, 0, 0}, r.tokens());
        assertEquals(StopReason.MAX_TOKENS, r.stopReason());
        assertEquals(1, stats.size());
        assertEquals(1, stats.get(0).rounds());
        assertEquals(2, stats.get(0).drafted());
        assertEquals(1, stats.get(0).accepted());
        assertArrayEquals(new int[]{1, 2, 3}, target.verifyCalls.get(0));
        assertEquals(Arrays.asList(1), target.rollbacks);
        assertEquals(4, target.memory().sequence().nPast());
    }
}

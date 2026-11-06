package com.eonmux.neuralfree.gguf.engine.draft;

import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.SpeculativeDecoder;
import com.eonmux.neuralfree.gguf.engine.StopReason;
import com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.FakeTarget;
import com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.PlainDrafter;
import com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.RoundMaker;
import com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.Run;
import com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.ScriptRound;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.logits;
import static com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.request;
import static com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.sampling;
import static com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.tokenizer;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.array;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.assertEq;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.cdfBreakpoints;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.dist;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.distSelected;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.emissionLaw;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.enumerate;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.greedyRow;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.measureConfirmed;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.minRatios;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.oneHotDrafter;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.residualWeights;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.round;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.thresholds;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.union;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * FLy deferral ({@link FlyVerifier}) on a 4-token vocabulary: a high-entropy rejection deferred behind natively
 * accepted future draft tokens, the entropy gate and the window that keep the rejection standing, greedy
 * target-only acceptance on the logit entropy, fly.py's top-k entropy, and both deferral forms through the
 * SpeculativeDecoder.
 */
public class FlyVerifierTest {

    /**
     * Non-greedy FLy defers a high-entropy rejection whose next {@code W} draft tokens are natively accepted: the
     * deferred draft token is emitted instead of the residual, and the residual only fires below the gate or
     * without the window.
     */
    @Test
    public void flyDefersAHighEntropyRejectionBehindNativeFutureAcceptances() {
        double[] peakedAccept = array(0.0, 0.0, 1.0, 0.0);  // row 0 confirms token 2 regardless of the uniform
        double[] highEntropyZero = array(0.3, 0.1, 0.3, 0.3); // row 1: p(1) > 0, top-3 entropy ln 3
        double[] p2 = array(0.0, 0.0, 1.0, 0.0);            // row 2: token 2 natively accepted
        double[] p3 = array(0.0, 0.0, 0.0, 1.0);            // row 3: token 3 natively accepted
        int[] draft = {2, 1, 2, 3};
        // the residual weights of row 1 against the one-hot proposal of token 1
        double[] w1 = residualWeights(highEntropyZero, array(0.0, 1.0, 0.0, 0.0));

        FlyVerifier deferring = new FlyVerifier(2, 0.3, 3, false);
        ScriptRound deferral = round(deferring, flyRound(peakedAccept, highEntropyZero, p2, p3, draft),
                new double[]{0.5, 0.9, 0.5, 0.5});
        // four pre-pass uniforms, and the deferred token 1 emitted like a confirmation.
        assertEquals(4, deferral.uniformsUsed);
        assertEquals(4, deferral.confirmed);
        assertEquals(Arrays.asList(2, 1, 2, 3, 1), deferral.emitted);

        // below the entropy gate the rejection stands and the residual carries the replacement.
        // u1 below the acceptance ratio 0.1 confirms the block natively; above it the gate holds
        // and the residual is enumerated.
        FlyVerifier gated = new FlyVerifier(2, 2.0, 3, false);
        double[] accept1 = minRatios(highEntropyZero, array(0.0, 1.0, 0.0, 0.0));
        List<Run> runs = enumerate(gated, flyRound(peakedAccept, highEntropyZero, p2, p3, draft),
                thresholds(union(accept1, cdfBreakpoints(w1))));
        assertEq(0.9, measureConfirmed(runs, 1), 1e-9, "gated deferral leaves the rejection standing");
        assertEq(0.1, measureConfirmed(runs, 4), 1e-9, "native acceptance below the ratio confirms the block");
        double[] gatedLaw = emissionLaw(runs);
        assertEq(0.1, gatedLaw[1], 1e-9, "the confirmed block's bonus is the chain draw");
        for (int x : new int[]{0, 2, 3}) {
            assertEq(0.3, gatedLaw[x], 1e-9, "fly residual law for token " + x);
        }

        // without the window room the deferral never fires either (k = 2 leaves no room for W = 2).
        FlyVerifier narrow = new FlyVerifier(2, 0.3, 3, false);
        List<Run> shortRuns = enumerate(narrow, flyRoundShort(peakedAccept, highEntropyZero, draft),
                thresholds(union(accept1, cdfBreakpoints(w1))));
        assertEq(0.1, measureConfirmed(shortRuns, 2), 1e-9, "native acceptance confirms both");
        assertEq(0.9, measureConfirmed(shortRuns, 1), 1e-9, "no window, no deferral");
        double[] shortLaw = emissionLaw(shortRuns);
        assertEq(0.1, shortLaw[1], 1e-9, "the confirmed block's bonus is the chain draw");
        for (int x : new int[]{0, 2, 3}) {
            assertEq(0.3, shortLaw[x], 1e-9, "no-window residual law for token " + x);
        }
    }

    private static RoundMaker flyRound(final double[] row0, final double[] row1, final double[] row2,
                                       final double[] row3, final int[] draft) {
        return new RoundMaker() {
            @Override public ScriptRound round(double[] uniforms) {
                Verifier.Distribution bonus = distSelected(1, 0.0, 1.0, 0.0, 0.0);
                return new ScriptRound(new Verifier.Distribution[]{dist(row0), dist(row1), dist(row2), dist(row3),
                        bonus}, new Verifier.Distribution[]{oneHotDrafter(draft[0]), oneHotDrafter(draft[1]),
                        oneHotDrafter(draft[2]), oneHotDrafter(draft[3])}, draft, uniforms);
            }
        };
    }

    private static RoundMaker flyRoundShort(final double[] row0, final double[] row1, final int[] draft) {
        return new RoundMaker() {
            @Override public ScriptRound round(double[] uniforms) {
                Verifier.Distribution bonus = distSelected(1, 0.0, 1.0, 0.0, 0.0);
                return new ScriptRound(new Verifier.Distribution[]{dist(row0), dist(row1), bonus},
                        new Verifier.Distribution[]{oneHotDrafter(draft[0]), oneHotDrafter(draft[1])},
                        new int[]{draft[0], draft[1]}, uniforms);
            }
        };
    }

    /** Greedy FLy uses target-only acceptance and defers on the softmax entropy of the logits. */
    @Test
    public void flyGreedyTargetOnlyAcceptanceDefersOnTheLogitEntropy() {
        // row 0: argmax 1, drafted 2 with a finite logit; row 1's argmax matches its draft.
        Verifier.Distribution[] targets = {greedyRow(1, 0.9, 2.0, 1.6, 0.5), greedyRow(3, 0.0, 0.0, 0.0, 2.0),
                greedyRow(0, 2.0, 0.0, 0.0, 0.0)};
        GreedyFlyRound greeter = new GreedyFlyRound(targets, new int[]{2, 3});
        ScriptRound deferral = round(new FlyVerifier(0, 0.3, 3, true), greeter, new double[0]);
        assertEquals(2, deferral.confirmed);
        assertEquals(Arrays.asList(2, 3, 0), deferral.emitted);
        assertEquals(0, deferral.uniformsUsed);

        // a low-entropy row decides against the deferral: the argmax replaces the draft.
        targets[0] = greedyRow(0, 5.0, Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY);
        ScriptRound rejected = round(new FlyVerifier(0, 0.3, 3, true), new GreedyFlyRound(targets, new int[]{2, 3}),
                new double[]{0.5});
        assertEquals(0, rejected.confirmed);
        assertEquals(Arrays.asList(0), rejected.emitted);

        // a native (argmax) match never defers: it simply confirms.
        targets[0] = greedyRow(2, 0.0, 0.0, 2.0, 0.0);
        ScriptRound nativeAccept = round(new FlyVerifier(0, 0.3, 3, true),
                new GreedyFlyRound(targets, new int[]{2, 3}), new double[0]);
        assertEquals(2, nativeAccept.confirmed);
        assertEquals(Arrays.asList(2, 3, 0), nativeAccept.emitted);
    }

    private static final class GreedyFlyRound implements RoundMaker {
        private final Verifier.Distribution[] targets;
        private final int[] draft;

        GreedyFlyRound(Verifier.Distribution[] targets, int[] draft) {
            this.targets = targets;
            this.draft = draft;
        }

        @Override public ScriptRound round(double[] uniforms) {
            Verifier.Distribution[] drafts = {greedyRow(draft[0], 1.0, 0.0, 0.0, 0.0),
                    greedyRow(draft[1], 0.0, 0.0, 0.0, 1.0)};
            return new ScriptRound(targets, drafts, draft, uniforms);
        }
    }

    /** fly.py's {@code compute_fly_entropy} over probabilities and over logits, with the top-k cap. */
    @Test
    public void flyEntropyFollowsTheTopKEntropyOfProcessedProbabilities() {
        double h = FlyVerifier.entropy(dist(0.25, 0.25, 0.25, 0.25), 3, false);
        // fly.py sums p·ln p over the top k without renormalizing
        assertEq(-3.0 * 0.25 * Math.log(0.25), h, 1e-9, "uniform top-3 entropy");
        assertEq(0.0, FlyVerifier.entropy(arrayDist(1.0), 3, false), 1e-9, "peaked entropy");
        // the logit form recurrently normalizes over the whole support.
        Verifier.Distribution row = greedyRow(0, Math.log(0.5), Math.log(0.25), Math.log(0.25), Math.log(0.0) + 0);
        double fromLogits = FlyVerifier.entropy(row, 3, true);
        double expected = -(0.5 * Math.log(0.5) + 2 * 0.25 * Math.log(0.25));
        assertEq(expected, fromLogits, 1e-9, "logit-form top-3 entropy");
        assertEq(0.0, FlyVerifier.entropy(greedyRow(1, 5.0, Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY,
                Float.NEGATIVE_INFINITY), 3, true), 1e-9, "spiky logits have no entropy");
    }

    private static Verifier.Distribution arrayDist(double... probs) {
        return distSelected(0, probs);
    }

    // ---------------------------------------------------------------- through the decoder

    /** Greedy FLy defers a mismatched draft when the entropy gate and the future argmaxes allow it. */
    @Test
    public void flyGreedyDeferralReplacesTheArgmaxOnAGatedMismatch() {
        // prefill argmax 1; row0 argmax 1 != d0 = 2 (deferral), row1 argmax 3 == d1, bonus row argmax 0.
        FakeTarget deferred = new FakeTarget(false)
                .add(logits(0.0, 1.0, 0.0, 0.0))
                .add(logits(0.9, 2.0, 1.6, 0.5), logits(0.0, 0.0, 0.0, 1.0), logits(1.0, 0.0, 0.0, 0.0));
        SpeculativeDecoder fly = SpeculativeDecoder.builder(deferred, new PlainDrafter(new int[]{2, 3}),
                tokenizer()).draftMax(2).verifyMode(VerifyMode.FLY).build();
        GenerationResult r = fly.generate(new int[]{0}, request(4, SamplerConfig.greedyDecoding()),
                (TokenCallback) null);
        assertArrayEquals(new int[]{1, 2, 3, 0}, r.tokens());
        assertTrue("the deferred round drops nothing", deferred.rollbacks.isEmpty());

        // with the gate closed the deferral stands down and the replacement is the argmax.
        FakeTarget gated = new FakeTarget(false)
                .add(logits(0.0, 1.0, 0.0, 0.0))
                .add(logits(0.0, 5.0, 0.0, 0.0), logits(0.0, 0.0, 0.0, 1.0), logits(1.0, 0.0, 0.0, 0.0));
        SpeculativeDecoder off = SpeculativeDecoder.builder(gated, new PlainDrafter(new int[]{2, 3}), tokenizer())
                .draftMax(2).verifyMode(VerifyMode.FLY).flyEntropyThreshold(2.0).build();
        GenerationResult plain = off.generate(new int[]{0}, request(2, SamplerConfig.greedyDecoding()),
                (TokenCallback) null);
        assertArrayEquals(new int[]{1, 1}, plain.tokens());
        assertEquals(StopReason.MAX_TOKENS, plain.stopReason());
        assertEquals(Arrays.asList(2), gated.rollbacks);
    }

    /** Non-greedy FLy deferral with probabilistic drafter probabilities is exercised at the verifier level; here
     * the one-hot drafter case confirms the pre-pass stream through the decoder. */
    @Test
    public void flyNonGreedyDeferralRunsThroughTheDecoder() {
        // row0 confirms d0; row1 natively rejects d1 (u = 0.16 > p(1) = 0.14) and defers behind rows 2-3.
        FakeTarget target = new FakeTarget(false)
                .add(logits(0.0, 1.0, 0.0, 0.0))
                .add(logits(0.0, 0.0, 1.0, 0.0), logits(0.28, 0.14, 0.29, 0.29),
                        logits(0.0, 0.0, 1.0, 0.0), logits(0.0, 0.0, 0.0, 1.0), logits(0.0, 1.0, 0.0, 0.0));
        SpeculativeDecoder dec = SpeculativeDecoder.builder(target, new PlainDrafter(new int[]{2, 1, 2, 3}),
                tokenizer()).draftMax(4).verifyMode(VerifyMode.FLY).flyWindow(2).build();
        GenerationResult r = dec.generate(new int[]{0}, request(5, sampling()), (TokenCallback) null);
        // 1, then the whole deferred block 2, 1, 2, 3: the budget ends the round at the last confirmation.
        assertArrayEquals(new int[]{1, 2, 1, 2, 3}, r.tokens());
        assertTrue(target.rollbacks.isEmpty());
        assertEquals(6, target.memory().sequence().nPast());
    }
}

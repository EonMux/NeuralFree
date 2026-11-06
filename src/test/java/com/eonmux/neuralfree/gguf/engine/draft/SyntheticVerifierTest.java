package com.eonmux.neuralfree.gguf.engine.draft;

import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.SpeculativeDecoder;
import com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.FakeTarget;
import com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.PlainDrafter;
import com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.RoundMaker;
import com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.Run;
import com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.ScriptRound;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

import static com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.SEED;
import static com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.logits;
import static com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.request;
import static com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.sampling;
import static com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.tokenizer;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.VOCAB;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.assertEq;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.dist;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.distSelected;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.emissionLaw;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.enumerate;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.measureConfirmed;
import static com.eonmux.neuralfree.gguf.engine.draft.VerifierEnumeration.oneHotDrafter;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Synthetic verification ({@link SyntheticVerifier}, vLLM's synthetic acceptance): the minimum-variance rates of an
 * acceptance length, rate validation, the realized per-position acceptance law by exact enumeration, the
 * {@code neuralfree.spec.synthetic} gate and the rate law through the SpeculativeDecoder.
 */
public class SyntheticVerifierTest {

    /** rates resolve from synthetic_acceptance_length with vLLM's minimum-variance schedule. */
    @Test
    public void syntheticAcceptanceLengthResolvesToTheMinimumVarianceRates() {
        assertArrayEquals(new double[]{1.0, 0.5, 0.0}, SyntheticVerifier.ratesForLength(2.5, 3), 0.0);
        assertArrayEquals(new double[]{0.75, 0.0, 0.0}, SyntheticVerifier.ratesForLength(1.75, 3), 0.0);
        assertArrayEquals(new double[]{0.0, 0.0, 0.0}, SyntheticVerifier.ratesForLength(1.0, 3), 0.0);
        assertArrayEquals(new double[]{1.0, 1.0, 1.0}, SyntheticVerifier.ratesForLength(4.0, 3), 0.0);
        assertThrows(IllegalArgumentException.class, () -> SyntheticVerifier.ratesForLength(0.0, 3));
        assertThrows(IllegalArgumentException.class, () -> SyntheticVerifier.ratesForLength(5.0, 3));
    }

    @Test
    public void syntheticVerifierValidatesItsRates() {
        SyntheticVerifier verifier = new SyntheticVerifier(new double[]{1.0, 0.5, 0.0});
        assertArrayEquals(new double[]{1.0, 0.5, 0.0}, verifier.rates(), 0.0);
        assertTrue(verifier.toString().contains("rates=[1.0, 0.5, 0.0]"));
        assertThrows(IllegalArgumentException.class, () -> new SyntheticVerifier(new double[0]));
        assertThrows(IllegalArgumentException.class, () -> new SyntheticVerifier(null));
        assertThrows(IllegalArgumentException.class, () -> new SyntheticVerifier(new double[]{0.5, 1.0}));
        assertThrows(IllegalArgumentException.class, () -> new SyntheticVerifier(new double[]{-0.1}));
        assertThrows(IllegalArgumentException.class, () -> new SyntheticVerifier(new double[]{1.1}));
    }

    /** The unconditional per-position rates: P(accepted >= j) exactly rates[j-1], with the chain-draw tokens. */
    @Test
    public void syntheticVerificationRealizesTheConfiguredRates() {
        SyntheticVerifier verifier = new SyntheticVerifier(new double[]{1.0, 0.5});
        List<Run> runs = enumerate(verifier, new RoundMaker() {
            @Override public ScriptRound round(double[] uniforms) {
                return new ScriptRound(
                        new Verifier.Distribution[]{dist(0.4, 0.3, 0.2, 0.1), dist(0.05, 0.25, 0.5, 0.2),
                                distSelected(3, 1.0, 0.0, 0.0, 0.0)},
                        new Verifier.Distribution[]{oneHotDrafter(2), oneHotDrafter(1)},
                        new int[]{2, 1}, uniforms);
            }
        }, new double[0], new double[]{0.5});
        assertEq(0.5, measureConfirmed(runs, 2), 1e-9, "first rate of 1.0 always accepts");
        assertEq(0.5, measureConfirmed(runs, 1), 1e-9, "second rate of 0.5 halves the block");
        // a rejection and the bonus both emit the chain's own draw, so the emission law stays the target's.
        double[] law = emissionLaw(runs);
        double[] expected = new double[VOCAB];
        expected[3] = 0.5;                                  // bonus after 2 confirms
        // rejects at position 1 draw target row 1's chain token (its argmax: token 2).
        expected[2] = 0.5;
        for (int x = 0; x < VOCAB; x++) {
            assertEq(expected[x], law[x], 1e-9, "synthetic emission law for token " + x);
        }
    }

    // ---------------------------------------------------------------- through the decoder

    @Test
    public void syntheticModeIsGatedBehindTheSystemProperty() {
        FakeTarget target = new FakeTarget(false);
        SpeculativeDecoder.Builder builder = SpeculativeDecoder.builder(target, new PlainDrafter(new int[]{2}),
                tokenizer()).draftMax(1).verifyMode(VerifyMode.SYNTHETIC).syntheticAcceptanceRates(0.5);
        IllegalStateException without = assertThrows(IllegalStateException.class, builder::build);
        assertTrue(without.getMessage().contains("neuralfree.spec.synthetic"));
        System.setProperty(SyntheticVerifier.PROP_ENABLED, "true");
        try {
            SpeculativeDecoder dec = builder.build();
            assertEquals(VerifyMode.SYNTHETIC, dec.verifyMode());
            // synthetic mode without rates or a length is refused even when enabled
            assertThrows(IllegalArgumentException.class, () -> SpeculativeDecoder
                    .builder(target, new PlainDrafter(new int[]{2}), tokenizer()).draftMax(1)
                    .verifyMode(VerifyMode.SYNTHETIC).build());
            // a synthetic setting outside synthetic mode is refused
            assertThrows(IllegalArgumentException.class, () -> SpeculativeDecoder
                    .builder(target, new PlainDrafter(new int[]{2}), tokenizer()).draftMax(1)
                    .verifyMode(VerifyMode.REJECTION).syntheticAcceptanceRates(0.5).build());
            assertThrows(IllegalArgumentException.class, () -> SpeculativeDecoder
                    .builder(target, new PlainDrafter(new int[]{2}), tokenizer()).draftMax(2)
                    .verifyMode(VerifyMode.REJECTION).syntheticAcceptanceRates(0.5)
                    .syntheticAcceptanceLength(1.5).build());
        } finally {
            System.clearProperty(SyntheticVerifier.PROP_ENABLED);
        }
    }

    /** Synthetic verification realizes the rates' acceptance law round by round (one uniform per round). */
    @Test
    public void syntheticDecoderRunsTheRateLaw() {
        FakeTarget target = new FakeTarget(false);
        target.add(logits(0.0, 1.0, 0.0, 0.0));
        // three rounds fit the budget; every round confirms iff its uniform <= 0.5, tokens come from one-hot rows.
        for (int round = 0; round < 4; round++) {
            target.add(logits(0.0, 0.0, 0.0, 1.0), logits(0.0, 0.0, 0.0, 1.0));
        }
        System.setProperty(SyntheticVerifier.PROP_ENABLED, "true");
        try {
            List<SpeculativeDecoder.Stats> stats = new ArrayList<>();
            SpeculativeDecoder dec = SpeculativeDecoder.builder(target,
                    new PlainDrafter(new int[]{3}, new int[]{3}, new int[]{3}, new int[]{3}),
                    tokenizer()).draftMax(1).verifyMode(VerifyMode.SYNTHETIC).syntheticAcceptanceRates(0.5)
                    .statsSink(stats::add).build();
            GenerationResult r = dec.generate(new int[]{0}, request(5, sampling()), (TokenCallback) null);
            assertArrayEquals(new int[]{1, 3, 3, 3, 3}, r.tokens());
            // a confirmed round also emits the bonus row's chain draw, so the budget ends the third round
            SplittableRandom expectedStream = new SplittableRandom(SEED);
            int expectedAccepted = 0;
            for (int round = 0; round < 3; round++) {
                if (expectedStream.nextDouble() <= 0.5) {
                    expectedAccepted++;
                }
            }
            assertEquals(3, stats.get(0).rounds());
            assertEquals(3, stats.get(0).drafted());
            assertEquals(expectedAccepted, stats.get(0).accepted());
        } finally {
            System.clearProperty(SyntheticVerifier.PROP_ENABLED);
        }
    }
}

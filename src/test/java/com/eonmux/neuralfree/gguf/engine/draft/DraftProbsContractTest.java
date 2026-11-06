package com.eonmux.neuralfree.gguf.engine.draft;

import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.SpeculativeDecoder;
import com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.FakeTarget;
import com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.ProbDrafter;
import com.eonmux.neuralfree.gguf.engine.watermark.WatermarkAlgorithm;
import com.eonmux.neuralfree.gguf.engine.watermark.WatermarkConfig;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.logits;
import static com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.request;
import static com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.sampling;
import static com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.tokenizer;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The {@link DraftProbs} contract under REJECTION verification: a probabilistic drafter's proposal rows decide the
 * acceptance, and a row that does not cover the drafted token with a finite positive probability fails the
 * generation.
 */
public class DraftProbsContractTest {

    /** A REJECTION decoder over one draft token per round. */
    private static SpeculativeDecoder probDecoder(FakeTarget target, ProbDrafter drafter) {
        return SpeculativeDecoder.builder(target, drafter, tokenizer())
                .draftMax(1).verifyMode(VerifyMode.REJECTION).build();
    }

    /** A drafter's proposal row decides the acceptance: {@code p(d) >= q(d)} confirms the draft token outright. */
    @Test
    public void probabilisticDraftersSupplyTheProposalDistributions() {
        FakeTarget target = new FakeTarget(false)
                .add(logits(0.0, 1.0, 0.0, 0.0))
                // the verify scores [token, draft]: the second row is the bonus row, never read because
                // the confirmation spends the budget
                .add(logits(0.3, 0.0, 0.65, 0.05), logits(0.3, 0.0, 0.65, 0.05));
        ProbDrafter drafter = ProbDrafter.with(new int[]{2}, new double[]{0.1, 0.2, 0.3, 0.4});
        // p(d)/q(d) = 0.65/0.3 > 1: the draft token is a sure confirmation.
        List<SpeculativeDecoder.Stats> stats = new ArrayList<>();
        SpeculativeDecoder sure = SpeculativeDecoder.builder(target, drafter, tokenizer())
                .draftMax(1).verifyMode(VerifyMode.REJECTION).statsSink(stats::add).build();
        GenerationResult r = sure.generate(new int[]{0}, request(2, sampling()), (TokenCallback) null);
        // the prefill's token 1, then the confirmed draft token 2, which spends the budget.
        assertArrayEquals(new int[]{1, 2}, r.tokens());
        assertEquals(1, stats.get(0).drafted());
        assertEquals(1, stats.get(0).accepted());
    }

    /**
     * The decoder refuses rows that do not cover the drafted token with a positive probability. A watermarked chain
     * draws every proposal, so a missing row is a fault there; an ordinary chain reads a null row as a one-hot
     * proposal.
     */
    @Test
    public void probabilisticDrafterFaultsFailTheGeneration() {
        FaultyTargetless f0 = drafterFault(false, true, false, false);
        assertTrue(f0.message().contains("no proposal probabilities"));
        FaultyTargetless f1 = drafterFault(true, false, false, false);
        assertTrue(f1.message().contains("proposal probability 0"));
        FaultyTargetless f2 = drafterFault(false, false, true, false);
        assertTrue(f2.message().contains("but offered"));
        FaultyTargetless f3 = drafterFault(false, false, false, true);
        assertTrue(f3.message().contains("NaN"));
    }

    /** The failure message of a generation over a faulty drafter. */
    private static final class FaultyTargetless {
        private final String message;

        FaultyTargetless(String message) {
            this.message = message;
        }

        String message() {
            return message;
        }
    }

    private static FaultyTargetless drafterFault(boolean zero, boolean nullRow, boolean shortRow, boolean nan) {
        FakeTarget target = new FakeTarget(false)
                .add(logits(0.0, 1.0, 0.0, 0.0))
                // the verify scores [token, draft]; the fault fires before the bonus row is read
                .add(logits(0.3, 0.0, 0.65, 0.05), logits(1.0, 0.0, 0.0, 0.0));
        ProbDrafter drafter = ProbDrafter.with(new int[]{2}, new double[]{0.1, 0.2, 0.3, 0.4});
        drafter.faultyZero = zero;
        drafter.faultyNull = nullRow;
        drafter.faultyShort = shortRow;
        drafter.faultyNaN = nan;
        SpeculativeDecoder dec = probDecoder(target, drafter);
        IllegalStateException e;
        try {
            SamplerConfig sampling = sampling().toBuilder().watermark(WatermarkConfig.builder(7L)
                    .algorithm(WatermarkAlgorithm.DUAL_KEY_GUMBEL).build()).build();
            dec.generate(new int[]{0}, request(2, sampling), (TokenCallback) null);
            throw new AssertionError("a faulty probability row must fail the generation");
        } catch (IllegalStateException caught) {
            e = caught;
        }
        return new FaultyTargetless(e.getMessage());
    }
}

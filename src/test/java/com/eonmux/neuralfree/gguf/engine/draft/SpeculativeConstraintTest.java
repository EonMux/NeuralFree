package com.eonmux.neuralfree.gguf.engine.draft;

import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.SpeculativeDecoder;
import com.eonmux.neuralfree.gguf.engine.StopReason;
import com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.FakeTarget;
import com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.PlainDrafter;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.logits;
import static com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.sampling;
import static com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.tokenizer;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * A grammar constraint under speculative decoding (vLLM {@code backend_xgrammar.py}: drafts validated against the
 * matcher and cut at the first token it rejects, every verify row masked by the grammar state its draft prefix
 * leaves, rejected draft tokens rolled back). Over the 4-token vocabulary {@code a b c d} with the grammar
 * {@code "a" ("bc" | "cb") "d"}:
 * <ul>
 *   <li>a draft is verified only up to its first token the grammar refuses;</li>
 *   <li>the verifiers that read every verify row before confirming any (block, FLy) see each row under the
 *       grammar state of the draft tokens before it, and a rejected draft token leaves the grammar again;</li>
 *   <li>every verification mode emits only text the grammar accepts, and the request's logit bias runs in the
 *       same chain.</li>
 * </ul>
 */
public class SpeculativeConstraintTest {

    private static final ConstraintSpec GRAMMAR = ConstraintSpec.gbnf("root ::= \"a\" (\"bc\" | \"cb\") \"d\"");
    private static final int A = 0;
    private static final int B = 1;
    private static final int C = 2;
    private static final int D = 3;

    private static GenerationRequest constrained(SamplerConfig config) {
        return GenerationRequest.builder().maxNewTokens(4).sampler(config).constraint(GRAMMAR).build();
    }

    private static SpeculativeDecoder decoder(FakeTarget target, PlainDrafter drafter, VerifyMode mode) {
        return SpeculativeDecoder.builder(target, drafter, tokenizer()).draftMax(3).verifyMode(mode).build();
    }

    private static String text(GenerationResult r) {
        StringBuilder out = new StringBuilder();
        for (int t : r.tokens()) {
            out.append((char) ('a' + t));
        }
        return out.toString();
    }

    /**
     * Block verification reads rows 0 and 1 before it confirms anything. Row 1 scores the token after
     * {@code a b}: its raw logits favour {@code b} (0.9), which the grammar forbids there. Masked under the state
     * after {@code a b}, the row is certain of {@code c}, so the block's joint ratio is one, both drafts are
     * confirmed and the bonus row (after {@code a b c}) can only give {@code d}. Read under the state after
     * {@code a} alone, the row would keep {@code b}, the block would fall back to one confirmed token and its
     * residual would emit the forbidden {@code b}.
     */
    @Test
    public void blockVerificationScoresEachRowUnderItsDraftPrefix() {
        FakeTarget target = new FakeTarget(false)
                .add(logits(0.25, 0.25, 0.25, 0.25))
                .add(logits(0, 1, 0, 0), logits(0, 0.9, 0.1, 0), logits(0.25, 0.25, 0.25, 0.25));
        PlainDrafter drafter = new PlainDrafter(new int[]{B, C});
        List<SpeculativeDecoder.Stats> stats = new ArrayList<>();
        SpeculativeDecoder dec = SpeculativeDecoder.builder(target, drafter, tokenizer()).draftMax(3)
                .verifyMode(VerifyMode.BLOCK).statsSink(stats::add).build();

        GenerationResult r = dec.generate(new int[]{A}, constrained(sampling()), (TokenCallback) null);

        assertEquals("abcd", text(r));
        assertEquals(StopReason.MAX_TOKENS, r.stopReason());
        assertEquals(2, stats.get(0).accepted());
        assertEquals("no rejected position to drop", 0, target.rollbacks.size());
    }

    /**
     * A partially confirmed block takes its draft tokens past the confirmed prefix back out of the grammar: with
     * row 0 uncertain between the two orders, a rejected {@code c} leaves the grammar after {@code a} (where the
     * residual is the other order's first token), and a rejected {@code b} after {@code a c} leaves only the
     * {@code b} the grammar demands there; across seeds the emitted text stays a sentence of the grammar.
     */
    @Test
    public void aRejectedDraftTokenLeavesTheGrammarAgain() {
        for (VerifyMode mode : new VerifyMode[]{VerifyMode.BLOCK, VerifyMode.FLY}) {
            for (long seed = 0; seed < 24; seed++) {
                FakeTarget target = new FakeTarget(false)
                        .add(logits(0.25, 0.25, 0.25, 0.25))
                        .add(logits(0, 0.5, 0.5, 0), logits(0.25, 0.25, 0.25, 0.25),
                                logits(0.25, 0.25, 0.25, 0.25))
                        .otherwise(logits(0.25, 0.25, 0.25, 0.25));
                SamplerConfig config = sampling().toBuilder().seed(seed).build();
                GenerationResult r = decoder(target, new PlainDrafter(new int[]{C, B}), mode)
                        .generate(new int[]{A}, constrained(config), (TokenCallback) null);
                String text = text(r);
                assertTrue(mode + " seed " + seed + " emitted '" + text + "'", text.matches("a(bc|cb)d"));
            }
        }
    }

    /**
     * vLLM's {@code validate_tokens}: a draft is cut before its first token the grammar refuses, so the verify call
     * scores only the admitted prefix. After {@code a}, the draft {@code b b c} keeps {@code b} alone (a second
     * {@code b} cannot follow {@code a b}); greedy match then confirms it and the row after {@code a b} gives the
     * only legal {@code c}.
     */
    @Test
    public void aDraftIsCutBeforeItsFirstTokenTheGrammarRefuses() {
        FakeTarget target = new FakeTarget(false)
                .add(logits(0.25, 0.25, 0.25, 0.25))
                .add(logits(0, 1, 0, 0), logits(0.25, 0.25, 0.25, 0.25))
                .add(logits(0.25, 0.25, 0.25, 0.25));
        PlainDrafter drafter = new PlainDrafter(new int[]{B, B, C});
        List<SpeculativeDecoder.Stats> stats = new ArrayList<>();
        SpeculativeDecoder dec = SpeculativeDecoder.builder(target, drafter, tokenizer()).draftMax(3)
                .statsSink(stats::add).build();

        GenerationResult r = dec.generate(new int[]{A}, constrained(SamplerConfig.greedyDecoding()),
                (TokenCallback) null);

        assertEquals("abcd", text(r));
        assertArrayEquals("the verify call scores the pending token and the admitted draft prefix",
                new int[]{A, B}, target.verifyCalls.get(0));
        assertEquals(1, stats.get(0).drafted());
        assertEquals(1, stats.get(0).accepted());
    }

    /**
     * The request's logit bias runs in the speculative chain beside the grammar: the row after {@code a} favours
     * {@code b} (0.6) over {@code c} (0.4), a +5 bias on {@code c} turns the greedy pick, so the drafted {@code b}
     * is rejected and the grammar then demands {@code b d}.
     */
    @Test
    public void theLogitBiasSteersTheSpeculativeChain() {
        FakeTarget target = new FakeTarget(false)
                .add(logits(0.25, 0.25, 0.25, 0.25))
                .add(logits(0, 0.6, 0.4, 0), logits(0.25, 0.25, 0.25, 0.25), logits(0.25, 0.25, 0.25, 0.25))
                .otherwise(logits(0.25, 0.25, 0.25, 0.25));
        GenerationRequest request = GenerationRequest.builder().maxNewTokens(4)
                .sampler(SamplerConfig.greedyDecoding()).constraint(GRAMMAR).logitBias(C, 5f).build();

        GenerationResult r = decoder(target, new PlainDrafter(new int[]{B, C}), VerifyMode.MATCH)
                .generate(new int[]{A}, request, (TokenCallback) null);

        assertEquals("acbd", text(r));
        assertEquals("the two drafts after the rejected b leave memory", List.of(2), target.rollbacks);
    }

    /** A draft whose first token the grammar refuses is not verified at all: the round decodes plainly. */
    @Test
    public void aDraftTheGrammarRefusesOutrightFallsBackToAPlainDecode() {
        FakeTarget target = new FakeTarget(false)
                .add(logits(0.25, 0.25, 0.25, 0.25))
                .add(logits(0, 1, 0, 0))
                .add(logits(0.25, 0.25, 0.25, 0.25))
                .add(logits(0.25, 0.25, 0.25, 0.25));
        PlainDrafter drafter = new PlainDrafter(new int[]{D, D}, new int[]{D}, new int[]{A});
        SpeculativeDecoder dec = decoder(target, drafter, VerifyMode.MATCH);

        GenerationResult r = dec.generate(new int[]{A}, constrained(SamplerConfig.greedyDecoding()),
                (TokenCallback) null);

        assertEquals("abcd", text(r));
        assertTrue("no proposal was admitted", target.verifyCalls.isEmpty());
        assertEquals(3, target.decodes.size());
    }

    /** Every verification mode emits only grammar sentences, whatever the proposals and the draws. */
    @Test
    public void everyVerificationModeEmitsOnlyGrammarSentences() {
        VerifyMode[] modes = {VerifyMode.MATCH, VerifyMode.REJECTION, VerifyMode.BLOCK, VerifyMode.FLY};
        int[][] proposals = {{B, C, D}, {C, B, D}, {B, B, D}, {C, C}, {D}};
        for (VerifyMode mode : modes) {
            for (int[] proposal : proposals) {
                for (long seed = 0; seed < 8; seed++) {
                    FakeTarget target = uniformTarget();
                    SamplerConfig config = sampling().toBuilder().seed(seed).build();
                    GenerationResult r = decoder(target, new PlainDrafter(proposal, proposal, proposal), mode)
                            .generate(new int[]{A}, constrained(config), (TokenCallback) null);
                    String text = text(r);
                    assertTrue(mode + " draft " + Arrays.toString(proposal) + " seed " + seed
                            + " emitted '" + text + "'", text.matches("a(bc|cb)d"));
                }
            }
        }
    }

    /** A target with uniform rows for every call the run makes. */
    private static FakeTarget uniformTarget() {
        return new FakeTarget(false).otherwise(logits(0.25, 0.25, 0.25, 0.25));
    }
}

package com.eonmux.neuralfree.gguf.engine.beam;

import com.eonmux.neuralfree.gguf.engine.StopReason;
import com.eonmux.neuralfree.gguf.engine.TokenScore;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.function.Supplier;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The search loop over a scripted stepper (vLLM's {@code _beam_search_step} and the pruning of
 * {@code instance_new_beams}): the greedy width-one path, an end-of-generation candidate completing a beam with
 * its log probability counted and dropped from the output, the width-{@code w} pruning by the penalized score in
 * stable order, the ranking flip a length penalty causes, {@code ignore_eos} sampling through end tokens while a
 * request stop token still ends, the allowed-id filter, and the cancellation, context and input guards. Every
 * step submits the live beams' full sequences at once, one row each.
 */
public final class BeamSearchTest {

    /** A capture row: (id, logprob) pairs, the sampled token first, best logprob first. */
    private static TokenScore[] row(Object... pairs) {
        TokenScore[] out = new TokenScore[pairs.length / 2];
        for (int i = 0; i < out.length; i++) {
            out[i] = new TokenScore((Integer) pairs[2 * i], (Float) pairs[2 * i + 1]);
        }
        return out;
    }

    /** The scripted stepper: every {@link #step} call replays one scored step's rows in beam order, and records
     *  the full sequences it was handed. */
    private static final class Script implements BeamSearch.Stepper {
        private final List<List<TokenScore[]>> steps = new ArrayList<>();
        private final List<int[]> seen = new ArrayList<>();
        private boolean pad = true;
        private int taken;

        Script step(TokenScore[]... rows) {
            steps.add(Arrays.asList(rows));
            return this;
        }

        /** Plays the planned rows back exactly: a step handing fewer rows than beams is a mismatched step. */
        Script exact() {
            this.pad = false;
            return this;
        }

        @Override
        public List<TokenScore[]> step(List<int[]> sequences) {
            seen.addAll(sequences);
            List<TokenScore[]> plan = steps.get(Math.min(taken++, steps.size() - 1));
            List<TokenScore[]> out = new ArrayList<>();
            for (int i = 0; i < sequences.size(); i++) {
                if (!pad && i >= plan.size()) {
                    break;
                }
                out.add(plan.get(Math.min(i, plan.size() - 1)));
            }
            return out;
        }
    }

    private static int[] prompt() {
        return new int[]{1, 2};
    }

    /** The settings under test: width one, four scored steps, the search's own end-of-generation handling off
     *  unless a test turns it on. */
    private static BeamSearch.Builder settings() {
        return BeamSearch.settings().beamWidth(1).maxSteps(4);
    }

    @Test
    public void widthOneFollowsTheBestTokensUntilTheBudgetRunsOut() {
        Script script = new Script().step(row(5, -0.2f, 9, -0.9f)).step(row(6, -0.3f, 4, -0.4f));
        BeamSearchOutput out = BeamSearch.run(settings().maxSteps(2).build(), prompt(), script);
        assertEquals(1, out.sequences().size());
        BeamSearchSequence best = out.sequences().get(0);
        assertArrayEquals(new int[]{5, 6}, best.tokens());
        assertEquals(StopReason.MAX_TOKENS, best.stopReason());
        assertEquals(-0.5, best.cumulativeLogprob(), 1e-6);
        assertEquals(2, best.stepCount());
        assertEquals(2, out.steps());
        assertEquals(2, out.promptTokens());
        // every step submits the live beams' full sequences, prompt included, one row per beam
        assertArrayEquals(new int[]{1, 2}, script.seen.get(0));
        assertArrayEquals(new int[]{1, 2, 5}, script.seen.get(1));
    }

    @Test
    public void anEndOfGenerationCandidateCompletesTheBeamAndDropsFromTheOutput() {
        Supplier<Script> script = () -> new Script().step(row(3, -0.5f));
        BeamSearchOutput out = BeamSearch.run(settings().maxSteps(4).endOfGenerationIds(Set.of(3))
                .renderer(tokens -> "decoded:" + tokens.length).build(), prompt(), script.get());
        assertEquals(1, out.steps());
        BeamSearchSequence best = out.sequences().get(0);
        assertEquals(StopReason.EOS, best.stopReason());
        assertArrayEquals(new int[]{3}, best.tokens());
        assertArrayEquals(new int[0], best.outputTokens(false));
        assertEquals(-0.5, best.cumulativeLogprob(), 1e-6);
        assertEquals("decoded:0", best.text());
        assertEquals(BeamSearchSequence.NO_STOP_TOKEN, best.stopTokenId());
    }

    @Test
    public void widthTwoExpandsEveryBeamAndPrunesToTheBestByScore() {
        Script script = new Script()
                .step(row(10, -0.2f, 20, -0.6f))
                .step(row(11, -0.1f, 21, -0.5f), row(12, -0.3f, 22, -0.4f));
        BeamSearchOutput out = BeamSearch.run(settings().beamWidth(2).maxSteps(2).build(), prompt(), script);
        assertEquals(2, out.sequences().size());
        // the four candidates score -0.3, -0.7, -0.9 and -1.0: the survivors are 10+11 and 10+21, and every
        // child of the weaker first beam lost the pruning
        assertArrayEquals(new int[]{10, 11}, out.sequences().get(0).tokens());
        assertArrayEquals(new int[]{10, 21}, out.sequences().get(1).tokens());
        assertEquals(-0.3, out.sequences().get(0).cumulativeLogprob(), 1e-6);
        assertEquals(-0.7, out.sequences().get(1).cumulativeLogprob(), 1e-6);
        for (BeamSearchSequence beam : out.sequences()) {
            for (int token : beam.tokens()) {
                assertTrue(token + " was pruned at step one", token != 12 && token != 22);
            }
        }
    }

    @Test
    public void everyStepsCaptureLeadsWithTheCandidateTheBeamTook() {
        Script script = new Script().step(row(10, -0.1f, 20, -0.5f));
        BeamSearchOutput out = BeamSearch.run(settings().beamWidth(2).maxSteps(1).build(), prompt(), script);
        assertEquals(2, out.sequences().size());
        assertArrayEquals(new int[]{10}, out.sequences().get(0).tokens());
        assertArrayEquals(new int[]{20}, out.sequences().get(1).tokens());
        TokenScore[] tookFirst = out.sequences().get(0).logprobs().get(0);
        TokenScore[] tookSecond = out.sequences().get(1).logprobs().get(0);
        assertEquals(10, tookFirst[0].tokenId());
        assertEquals(-0.1f, tookFirst[0].logprob(), 1e-6);
        assertEquals(20, tookSecond[0].tokenId());
        assertEquals(-0.5f, tookSecond[0].logprob(), 1e-6);
        assertEquals(2, tookSecond.length);
    }

    @Test
    public void theLengthPenaltyFlipsTheFinalRanking() {
        // one beam completes on the eos at cum -0.4 over one token; the other grows four tokens to cum -1.0
        Supplier<Script> script = () -> new Script()
                .step(row(3, -0.4f, 30, -0.6f))
                .step(row(30, -0.4f / 3f))
                .step(row(30, -0.4f / 3f))
                .step(row(30, -0.4f / 3f));
        BeamSearchOutput longFirst = BeamSearch.run(settings().maxSteps(4).lengthPenalty(1.0f)
                .endOfGenerationIds(Set.of(3)).build(), prompt(), script.get());
        assertEquals(-0.25, longFirst.sequences().get(0).score(1.0f), 1e-6);
        assertArrayEquals(new int[]{30, 30, 30, 30}, longFirst.sequences().get(0).tokens());
        BeamSearchOutput shortFirst = BeamSearch.run(settings().maxSteps(4).lengthPenalty(0.0f)
                .endOfGenerationIds(Set.of(3)).build(), prompt(), script.get());
        assertArrayEquals(new int[]{3}, shortFirst.sequences().get(0).tokens());
        assertEquals(-0.4, shortFirst.sequences().get(0).cumulativeLogprob(), 1e-6);
    }

    @Test
    public void ignoreEosSamplesThroughEndTokensAndRunsTheBudgetOut() {
        Script script = new Script().step(row(3, -0.1f, 5, -0.2f)).step(row(3, -0.1f));
        BeamSearchOutput out = BeamSearch.run(settings().maxSteps(2).ignoreEos(true)
                .endOfGenerationIds(Set.of(3)).build(), prompt(), script);
        BeamSearchSequence best = out.sequences().get(0);
        assertEquals(StopReason.MAX_TOKENS, best.stopReason());
        assertArrayEquals(new int[]{3, 3}, best.tokens());
        assertEquals(-0.2, best.cumulativeLogprob(), 1e-6);
    }

    @Test
    public void aRequestStopTokenCompletesEvenUnderIgnoreEos() {
        Script script = new Script().step(row(7, -0.2f, 5, -0.9f));
        BeamSearchOutput out = BeamSearch.run(settings().maxSteps(4).ignoreEos(true)
                .stopTokenIds(new int[]{7}).build(), prompt(), script);
        BeamSearchSequence best = out.sequences().get(0);
        assertEquals(StopReason.EOS, best.stopReason());
        assertEquals(7, best.stopTokenId());
        assertArrayEquals(new int[0], best.outputTokens(false));
    }

    @Test
    public void theAllowedIdsFilterOutEveryOtherCandidate() {
        Script script = new Script().step(row(9, -0.1f, 4, -0.8f));
        BeamSearchOutput out = BeamSearch.run(settings().maxSteps(1).allowedTokenIds(new int[]{4}).build(),
                prompt(), script);
        assertArrayEquals(new int[]{4}, out.sequences().get(0).tokens());
    }

    @Test
    public void aTieStaysInCaptureOrder() {
        Script script = new Script().step(row(9, -0.5f, 4, -0.5f));
        BeamSearchOutput out = BeamSearch.run(settings().maxSteps(1).build(), prompt(), script);
        assertArrayEquals(new int[]{9}, out.sequences().get(0).tokens());
    }

    @Test
    public void aCancelledSearchEndsItsBeamsBeforeTheFirstStep() {
        Script script = new Script().step(row(5, -0.2f));
        BeamSearchOutput out = BeamSearch.run(settings().cancelled(() -> true).build(), prompt(), script);
        assertEquals(0, out.steps());
        assertEquals(1, out.sequences().size());
        assertEquals(StopReason.CANCELLED, out.sequences().get(0).stopReason());
        assertEquals(0, out.sequences().get(0).tokens().length);
    }

    @Test
    public void aFullContextEndsTheGrowingBeams() {
        // room for the prompt plus one more token: the second step would overflow, so it never runs
        Script script = new Script().step(row(5, -0.2f)).step(row(6, -0.2f));
        BeamSearchOutput out = BeamSearch.run(settings().maxSteps(4).maxSequenceLength(3).build(), prompt(),
                script);
        assertEquals(1, out.steps());
        BeamSearchSequence best = out.sequences().get(0);
        assertEquals(StopReason.CONTEXT_FULL, best.stopReason());
        assertArrayEquals(new int[]{5}, best.tokens());
    }

    @Test
    public void thePromptMustBeNonEmptyAndNonNegative() {
        Script script = new Script().step(row(5, -0.2f));
        assertThrows(IllegalArgumentException.class,
                () -> BeamSearch.run(settings().build(), new int[0], script));
        assertThrows(IllegalArgumentException.class,
                () -> BeamSearch.run(settings().build(), new int[]{1, -2}, script));
    }

    @Test
    public void aStepThatReturnsTheWrongRowsOrAnEmptyRowAborts() {
        Script missing = new Script().step(row(5, -0.2f, 6, -0.3f)).exact();
        IllegalStateException fewer = assertThrows(IllegalStateException.class,
                () -> BeamSearch.run(settings().beamWidth(2).build(), prompt(), missing));
        assertTrue(fewer.getMessage(), fewer.getMessage().contains("rows for 2 beams"));

        Script empty = new Script().step(row());
        IllegalStateException noRow = assertThrows(IllegalStateException.class,
                () -> BeamSearch.run(settings().build(), prompt(), empty));
        assertTrue(noRow.getMessage(), noRow.getMessage().contains("empty"));
    }

    @Test
    public void theSettingsValidateTheirBounds() {
        assertThrows(IllegalArgumentException.class, () -> BeamSearch.settings().beamWidth(0));
        assertThrows(IllegalArgumentException.class, () -> BeamSearch.settings().maxSteps(-1));
        assertThrows(IllegalArgumentException.class, () -> BeamSearch.settings().lengthPenalty(Float.NaN));
        assertThrows(IllegalArgumentException.class,
                () -> BeamSearch.settings().lengthPenalty(Float.NEGATIVE_INFINITY));
        assertThrows(IllegalArgumentException.class, () -> BeamSearch.settings().maxSequenceLength(0));
    }

    @Test
    public void withoutARendererTheTextStaysNull() {
        Script script = new Script().step(row(5, -0.2f));
        BeamSearchOutput out = BeamSearch.run(settings().maxSteps(1).build(), prompt(), script);
        assertNull(out.sequences().get(0).text());
    }

    @Test
    public void theOutputKeepsTheBestBeamsInRankingOrderAndTheListIsReadonly() {
        Script script = new Script()
                .step(row(10, -0.2f, 20, -0.6f))
                .step(row(11, -0.1f), row(21, -0.4f));
        BeamSearchOutput out = BeamSearch.run(settings().beamWidth(2).maxSteps(2).build(), prompt(), script);
        assertEquals(-0.3, out.sequences().get(0).cumulativeLogprob(), 1e-6);
        assertEquals(-1.0, out.sequences().get(1).cumulativeLogprob(), 1e-6);
        assertThrows(UnsupportedOperationException.class, () -> out.sequences().add(out.sequences().get(0)));
        assertThrows(UnsupportedOperationException.class,
                () -> out.sequences().get(0).logprobs().add(new TokenScore[0]));
    }
}

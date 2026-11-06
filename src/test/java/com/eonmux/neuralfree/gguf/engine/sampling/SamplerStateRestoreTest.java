package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.engine.Sampler;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.gguf.grammar.GbnfBuilder;
import com.eonmux.neuralfree.gguf.grammar.Grammar;
import com.eonmux.neuralfree.gguf.grammar.GrammarTriggers;
import org.junit.Test;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * {@link Sampler#saveState} and {@link Sampler#restoreState} over whole chains, as speculative verification uses
 * them: a sampler that ran rows and accepted tokens ahead of a save and restored it scores the next row exactly as a
 * twin that never went ahead (penalties with an evicting window, DRY, min-tokens, bad-words and a grammar in one
 * chain, a trigger-gated grammar, the reasoning budget, mirostat and adaptive-p), a state restores any number of
 * times, a control action after the save survives the restore, and adaptive-p keeps the selection of the row it was
 * saved after.
 */
public class SamplerStateRestoreTest {

    /** Byte vocabulary: id {@code c} renders the byte {@code c}; id 0 is end-of-generation with an empty piece. */
    private static final class ByteVocab implements SamplerVocab {
        @Override public int vocabSize() { return 128; }
        @Override public boolean isEog(int id) { return id == 0; }
        @Override public int eotToken() { return 0; }
        @Override public int eosToken() { return 0; }
        @Override public byte[] piece(int id) { return id <= 0 ? new byte[0] : new byte[]{(byte) id}; }
        @Override public String pieceText(int id) { return new String(piece(id), StandardCharsets.UTF_8); }
        @Override public int[] tokenize(String text) { return text.chars().toArray(); }
    }

    private static final ByteVocab VOCAB = new ByteVocab();

    /** A row where end-of-generation, {@code a} and {@code b} lead and every other id has a distinct logit. */
    private static float[] row() {
        float[] logits = new float[VOCAB.vocabSize()];
        for (int i = 0; i < logits.length; i++) {
            logits[i] = (i % 7) * 0.25f;
        }
        logits[0] = 2.5f;
        logits['a'] = 2.0f;
        logits['b'] = 1.9f;
        return logits;
    }

    /** The processed logit per token id of one chain run ({@code -inf} for a candidate the run dropped). */
    private static float[] scored(Sampler sampler, float[] row) {
        Sampler.Candidates cur = sampler.apply(row);
        float[] byId = new float[row.length];
        Arrays.fill(byId, Float.NEGATIVE_INFINITY);
        for (int i = 0; i < cur.size(); i++) {
            byId[cur.id(i)] = cur.logit(i);
        }
        return byId;
    }

    /** Runs the chain over {@code row} and accepts each token of {@code text} after it, as a decode loop does. */
    private static void feed(Sampler sampler, String text) {
        for (int token : text.chars().toArray()) {
            sampler.apply(row());
            sampler.accept(token);
        }
    }

    private static Grammar grammar(String gbnf) {
        return GbnfBuilder.grammarOf(ConstraintSpec.gbnf(gbnf));
    }

    /** Greedy penalties + DRY with min-tokens, bad-words and a grammar: the chain whose rows depend on history. */
    private static Sampler historyChain() {
        SamplerConfig config = SamplerConfig.greedyDecoding().toBuilder()
                .repeatPenalty(1.3f).presencePenalty(0.5f).frequencyPenalty(0.25f).penaltyLastN(3)
                .dryMultiplier(0.8f).dryBase(1.75f).dryAllowedLength(1).dryPenaltyLastN(8)
                .build();
        List<SamplerStage> extra = new ArrayList<>();
        extra.add(new GrammarStage(grammar("root ::= (\"a\" | \"b\")+"), VOCAB));
        extra.add(new MinTokensStage(6, VOCAB));
        extra.add(new BadWordsStage(Collections.singletonList(new int[]{'b', 'a', 'b'}), VOCAB));
        return Sampler.create(config, new int[0], extra, VOCAB);
    }

    @Test
    public void aRestoredChainScoresTheNextRowAsATwinThatNeverWentAhead() {
        Sampler ahead = historyChain();
        Sampler twin = historyChain();
        feed(ahead, "abba");
        feed(twin, "abba");
        Object saved = ahead.saveState();
        feed(ahead, "baabb");
        assertFalse("the tokens accepted ahead change the next row",
                Arrays.equals(scored(twin, row()), scored(ahead, row())));

        ahead.restoreState(saved);
        assertArrayEquals("after the restore", scored(twin, row()), scored(ahead, row()), 0f);
        feed(ahead, "ab");
        feed(twin, "ab");
        assertArrayEquals("the restored chain keeps tracking the twin", scored(twin, row()), scored(ahead, row()),
                0f);
    }

    @Test
    public void aStateRestoresAnyNumberOfTimes() {
        Sampler ahead = historyChain();
        Sampler twin = historyChain();
        feed(ahead, "ab");
        feed(twin, "ab");
        Object saved = ahead.saveState();
        for (String detour : new String[]{"bab", "aaaa", "b"}) {
            feed(ahead, detour);
            ahead.restoreState(saved);
            assertArrayEquals("after the detour " + detour, scored(twin, row()), scored(ahead, row()), 0f);
        }
    }

    @Test
    public void aGrammarRestoreRollsTheParseBack() {
        Sampler sampler = Sampler.create(SamplerConfig.greedyDecoding(), new int[0],
                Collections.<SamplerStage>singletonList(new GrammarStage(grammar("root ::= \"ab\" \"c\""), VOCAB)),
                VOCAB);
        feed(sampler, "a");
        Object afterA = sampler.saveState();
        feed(sampler, "bc");
        assertEquals("the completed parse admits only end-of-generation", 0, sampler.sample(row()));
        sampler.restoreState(afterA);
        float[] next = scored(sampler, row());
        assertTrue("after the rollback only b continues the parse", Float.isFinite(next['b']));
        assertEquals(Float.NEGATIVE_INFINITY, next['c'], 0f);
        assertEquals(Float.NEGATIVE_INFINITY, next[0], 0f);
    }

    private static Sampler triggered() {
        TriggeredGrammarStage stage = new TriggeredGrammarStage(grammar("root ::= \"<t>\" \"a\"+"),
                GrammarTriggers.lazyWords(Collections.singletonList("<t>")), VOCAB);
        return Sampler.create(SamplerConfig.greedyDecoding(), new int[0],
                Collections.<SamplerStage>singletonList(stage), VOCAB);
    }

    @Test
    public void aTriggerGatedGrammarRestoresItsGateAndItsParse() {
        Sampler ahead = triggered();
        Sampler twin = triggered();
        feed(ahead, "x<");
        feed(twin, "x<");
        Object waiting = ahead.saveState();
        feed(ahead, "t>a");
        assertFalse("the opened grammar masks free text",
                Arrays.equals(scored(twin, row()), scored(ahead, row())));
        ahead.restoreState(waiting);
        assertArrayEquals("the gate waits again with its trigger buffer", scored(twin, row()),
                scored(ahead, row()), 0f);

        feed(ahead, "t>a");
        feed(twin, "t>a");
        Object open = ahead.saveState();
        feed(ahead, "aa");
        ahead.restoreState(open);
        assertArrayEquals("the opened grammar's parse rolls back", scored(twin, row()), scored(ahead, row()), 0f);
    }

    private static ReasoningBudgetStage budget(int tokens) {
        GenerationRequest request = GenerationRequest.builder().maxNewTokens(64).reasoningBudget(tokens).build();
        return ReasoningBudgetStage.of(ReasoningFilter.thinkTags(), request, VOCAB);
    }

    private static Sampler over(SamplerStage stage) {
        return Sampler.create(SamplerConfig.greedyDecoding(), new int[0],
                Collections.singletonList(stage), VOCAB);
    }

    @Test
    public void theReasoningBudgetRestoresItsCountdown() {
        ReasoningBudgetStage stage = budget(3);
        Sampler sampler = over(stage);
        feed(sampler, "<think>");
        assertEquals(ReasoningBudgetStage.State.COUNTING, stage.state());
        Object counting = sampler.saveState();
        feed(sampler, "xyz");
        assertEquals("the budget ran out ahead", ReasoningBudgetStage.State.FORCING, stage.state());
        sampler.restoreState(counting);
        assertEquals(ReasoningBudgetStage.State.COUNTING, stage.state());
        feed(sampler, "xy");
        assertEquals("the restored countdown still has its full budget", ReasoningBudgetStage.State.COUNTING,
                stage.state());
        feed(sampler, "z");
        assertEquals(ReasoningBudgetStage.State.FORCING, stage.state());
    }

    @Test
    public void aForceAfterTheSaveSurvivesTheRestore() {
        ReasoningBudgetStage stage = budget(50);
        Sampler sampler = over(stage);
        feed(sampler, "<think>");
        Object counting = sampler.saveState();
        feed(sampler, "x");
        assertTrue(stage.force());
        sampler.restoreState(counting);
        assertEquals("the mid-stream control is never undone", ReasoningBudgetStage.State.FORCING, stage.state());
    }

    @Test
    public void mirostatRestoresItsSurpriseTarget() {
        SamplerConfig config = SamplerConfig.builder().temperature(1f).mirostat(2).mirostatTau(3f)
                .mirostatEta(0.5f).seed(7).build();
        assertRestoresLikeTheTwin(config);
    }

    @Test
    public void adaptivePRestoresItsAverage() {
        SamplerConfig config = SamplerConfig.builder().temperature(1f).samplers("temperature;adaptive_p")
                .adaptiveTarget(0.4f).adaptiveDecay(0.8f).seed(7).build();
        assertRestoresLikeTheTwin(config);
    }

    /**
     * Twin samplers draw the same tokens; one goes ahead and restores. The draws themselves continue on its random
     * stream, so the comparison is over the processed set, which the stage's restored state alone decides.
     */
    private static void assertRestoresLikeTheTwin(SamplerConfig config) {
        Sampler ahead = Sampler.create(config);
        Sampler twin = Sampler.create(config);
        for (int i = 0; i < 4; i++) {
            ahead.accept(ahead.sample(row()));
            twin.accept(twin.sample(row()));
        }
        Object saved = ahead.saveState();
        for (int i = 0; i < 5; i++) {
            ahead.accept(ahead.sample(row()));
        }
        ahead.restoreState(saved);
        assertArrayEquals(scored(twin, row()), scored(ahead, row()), 0f);
    }

    @Test
    public void adaptivePKeepsTheSelectionOfTheRowItWasSavedAfter() {
        SamplerConfig config = SamplerConfig.builder().temperature(1f).samplers("temperature;adaptive_p")
                .adaptiveTarget(0.4f).adaptiveDecay(0.8f).seed(7).build();
        Sampler ahead = Sampler.create(config);
        Sampler twin = Sampler.create(config);
        int selected = ahead.sample(row());
        assertEquals(selected, twin.sample(row()));
        Object saved = ahead.saveState();
        ahead.accept(selected);
        ahead.restoreState(saved);
        ahead.accept(selected);
        twin.accept(selected);
        assertArrayEquals("the accept after the restore folds the saved selection into the average",
                scored(twin, row()), scored(ahead, row()), 0f);
    }
}

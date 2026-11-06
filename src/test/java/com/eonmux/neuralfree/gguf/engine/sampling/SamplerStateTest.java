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
import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The chain's token-history state ({@link Sampler#saveState}, {@link SamplerStage#saveState}) and its constraint
 * check ({@link Sampler#admittedPrefix}, vLLM's {@code validate_tokens}): for every stage that records accepted
 * tokens, a chain that saved its state, accepted more tokens and restored the state scores every row exactly as a
 * chain that never saw those tokens, and keeps doing so as both accept the same tokens afterwards; the constraint
 * check walks a token sequence through the grammars and leaves them as they were. The terminal selectors that adapt
 * on the accepted token's row (mirostat's {@code mu}, adaptive-p's average) adapt on accept, not on apply, and keep
 * their pending row in the state, so speculative verification's read-ahead rows leave no trace.
 */
public class SamplerStateTest {

    /** Byte tokens: id {@code c} renders the byte {@code c}; id 0 is EOG (no piece). */
    private static final class ByteVocab implements SamplerVocab {
        @Override public int vocabSize() { return 128; }
        @Override public boolean isEog(int id) { return id == 0; }
        @Override public int eotToken() { return 0; }
        @Override public int eosToken() { return 0; }
        @Override public byte[] piece(int id) { return id <= 0 ? new byte[0] : new byte[]{(byte) id}; }
        @Override public String pieceText(int id) { return new String(piece(id), StandardCharsets.UTF_8); }
        @Override public int[] tokenize(String text) { return chars(text); }
    }

    private static final ByteVocab VOCAB = new ByteVocab();

    private static int[] chars(String text) {
        return text.chars().toArray();
    }

    private static Grammar grammar(String gbnf) {
        return GbnfBuilder.grammarOf(ConstraintSpec.gbnf(gbnf));
    }

    /** Distinct logits over the whole vocabulary, EOG included, so every mask and penalty shows. */
    private static float[] probe(int salt) {
        float[] logits = new float[VOCAB.vocabSize()];
        for (int i = 0; i < logits.length; i++) {
            logits[i] = 1f + ((i * 37 + salt * 11) % 101) / 25f;
        }
        return logits;
    }

    private static Sampler chainOf(SamplerStage stage) {
        return Sampler.create(SamplerConfig.greedyDecoding(), new int[0], List.of(stage), VOCAB);
    }

    /** The post-chain logit of every token id after one apply. */
    private static float[] scores(Sampler sampler, float[] logits) {
        Sampler.Candidates cur = sampler.apply(logits);
        float[] byId = new float[logits.length];
        Arrays.fill(byId, Float.NaN);
        for (int i = 0; i < cur.size(); i++) {
            byId[cur.id(i)] = cur.logit(i);
        }
        return byId;
    }

    /**
     * Two chains over fresh stages from {@code factory} take {@code prefix}; one saves, accepts {@code extra} (which
     * must change its scores), restores, and must then score like the other, before and after both accept
     * {@code after}.
     */
    private static void assertRestores(Supplier<SamplerStage> factory, int[] prefix, int[] extra, int[] after) {
        Sampler restored = chainOf(factory.get());
        Sampler reference = chainOf(factory.get());
        for (Sampler s : new Sampler[]{restored, reference}) {
            s.apply(probe(0)); // arms the stages that count generated tokens only
            s.accept(prefix, 0, prefix.length);
        }
        Object state = restored.saveState();
        restored.accept(extra, 0, extra.length);
        assertFalse("the extra tokens must change the scores for the check to mean anything",
                Arrays.equals(scores(restored, probe(1)), scores(reference, probe(1))));
        restored.restoreState(state);
        assertArrayEquals(scores(reference, probe(2)), scores(restored, probe(2)), 0f);
        restored.accept(after, 0, after.length);
        reference.accept(after, 0, after.length);
        assertArrayEquals(scores(reference, probe(3)), scores(restored, probe(3)), 0f);
        restored.restoreState(state);
        reference.reset();
        reference.apply(probe(0));
        reference.accept(prefix, 0, prefix.length);
        assertArrayEquals("a state restores more than once", scores(reference, probe(4)),
                scores(restored, probe(4)), 0f);
    }

    @Test
    public void thePenaltyWindowRestoresWithTheTokensItHadEvicted() {
        assertRestores(() -> new PenaltiesStage(1.3f, 0.2f, 0.1f, 4, true), new int[]{1, 2, 3, 4, 5},
                new int[]{6, 6, 7}, new int[]{8});
    }

    @Test
    public void theDryHistoryRestores() {
        assertRestores(() -> new DryStage(0.8f, 1.75f, 2, 16, List.of(new int[]{'\n'})), chars("abcab"),
                chars("cab"), chars("c"));
    }

    @Test
    public void theGrammarStacksRestore() {
        assertRestores(() -> new GrammarStage(grammar("root ::= \"a\" [a-c]* \"x\" \"yz\""), VOCAB), chars("ab"),
                chars("cxy"), chars("c"));
    }

    @Test
    public void theMinTokensCountRestores() {
        assertRestores(() -> new MinTokensStage(3, VOCAB), chars("a"), chars("bc"), chars("d"));
    }

    @Test
    public void theBadWordsTailRestores() {
        assertRestores(() -> new BadWordsStage(List.of(chars("abc"), chars("q")), VOCAB), chars("xa"), chars("b"),
                chars("a"));
    }

    @Test
    public void theTriggeredGrammarRestoresToFreeTextBeforeItsTrigger() {
        GrammarTriggers triggers = GrammarTriggers.lazyWords(List.of("<t>"));
        Grammar g = grammar("root ::= \"<t>\" [x-z]+");
        assertRestores(() -> new TriggeredGrammarStage(g, triggers, VOCAB), chars("hi <"), chars("t>"),
                chars("t>x"));
    }

    @Test
    public void theReasoningBudgetRestoresItsCountdown() {
        GenerationRequest budget = GenerationRequest.builder().maxNewTokens(64).reasoningBudget(2).build();
        assertRestores(() -> ReasoningBudgetStage.of(ReasoningFilter.thinkTags(), budget, VOCAB),
                chars("<think>x"), chars("yz"), chars("w"));
    }

    @Test
    public void theAdmittedPrefixWalksTheGrammarWithoutChangingIt() {
        Sampler sampler = chainOf(new GrammarStage(grammar("root ::= \"ab\" (\"c\" | \"d\")"), VOCAB));
        assertEquals('a', sampler.sample(probe(5)));
        sampler.accept('a');
        assertEquals("a completed sentence admits end-of-generation", 3,
                sampler.admittedPrefix(new int[]{'b', 'c', 0}));
        assertEquals(2, sampler.admittedPrefix(new int[]{'b', 'd', 'd'}));
        assertEquals(1, sampler.admittedPrefix(new int[]{'b', 'b'}));
        assertEquals(0, sampler.admittedPrefix(new int[]{0}));
        assertEquals(0, sampler.admittedPrefix(new int[]{'x', 'b'}));
        float[] after = scores(sampler, probe(6));
        for (int id = 0; id < after.length; id++) {
            assertEquals("the walk left the grammar after 'a' (token " + id + ")", id == 'b',
                    after[id] != Float.NEGATIVE_INFINITY);
        }
    }

    @Test
    public void theAdmittedPrefixFollowsATriggeredGrammarAcrossItsTrigger() {
        GrammarTriggers triggers = GrammarTriggers.lazyWords(List.of("<t>"));
        Sampler sampler = chainOf(new TriggeredGrammarStage(grammar("root ::= \"<t>\" [x-z]+"), triggers, VOCAB));
        sampler.apply(probe(0));
        assertEquals("free text passes", 4, sampler.admittedPrefix(chars("hey ")));
        assertEquals("the fired grammar refuses 'a'", 5, sampler.admittedPrefix(chars("<t>xya")));
        sampler.accept(chars("<t>"), 0, 3);
        assertEquals(0, sampler.admittedPrefix(chars("a")));
        assertEquals(2, sampler.admittedPrefix(chars("zz")));
    }

    @Test
    public void aChainWithoutAConstraintAdmitsEverything() {
        Sampler sampler = chainOf(new MinTokensStage(2, VOCAB));
        sampler.apply(probe(0));
        assertEquals(3, sampler.admittedPrefix(new int[]{0, 0, 0}));
        assertThrows(IllegalArgumentException.class,
                () -> chainOf(new GrammarStage(grammar("root ::= \"a\""), VOCAB)).admittedPrefix(new int[]{-1}));
    }

    @Test
    public void aStateOfAnotherSamplerIsRefused() {
        Sampler one = chainOf(new MinTokensStage(2, VOCAB));
        Sampler two = chainOf(new MinTokensStage(2, VOCAB));
        Object state = one.saveState();
        assertThrows(IllegalArgumentException.class, () -> two.restoreState(state));
        assertThrows(IllegalArgumentException.class, () -> two.restoreState("state"));
        assertNotEquals(null, state);
        assertTrue(new GrammarStage(grammar("root ::= \"a\""), VOCAB).constrains());
        assertFalse(new MinTokensStage(2, VOCAB).constrains());
    }

    private static SamplerConfig.Builder plain() {
        return SamplerConfig.builder().topK(0).topP(1f).minP(0f).typicalP(1f).temperature(1f).seed(7);
    }

    private static Sampler sampler(SamplerConfig config) {
        return Sampler.create(config, new int[0], VOCAB);
    }

    /**
     * A Zipf row over the vocabulary ({@code p ~ rank^-1.2}) with the ranks permuted by {@code salt}: steep enough
     * that both mirostat versions truncate it at a {@code mu} near 6.
     */
    private static float[] zipf(int salt) {
        float[] logits = new float[VOCAB.vocabSize()];
        for (int rank = 0; rank < logits.length; rank++) {
            logits[(rank * 37 + salt * 11) % logits.length] = (float) (-1.2 * Math.log(rank + 1));
        }
        return logits;
    }

    /** A row certain of one token (picked by {@code salt}): mirostat keeps it alone, at surprise 0. */
    private static float[] certain(int salt) {
        float[] logits = zipf(salt);
        logits[(salt * 11) % logits.length] = 40f;
        return logits;
    }

    /** The final candidate set of one apply: id and probability per candidate, in the set's order. */
    private static float[] candidateSet(Sampler sampler, float[] logits) {
        Sampler.Candidates cur = sampler.apply(logits);
        float[] out = new float[2 * cur.size()];
        for (int i = 0; i < cur.size(); i++) {
            out[2 * i] = cur.id(i);
            out[2 * i + 1] = cur.prob(i);
        }
        return out;
    }

    /**
     * llama.cpp adapts mirostat's {@code mu} inside apply by the selected token's surprise; here apply and the
     * accept of the selected token together do the same. Over a uniform five-token row (surprise 2.32193 each) with
     * {@code tau} 1.2, {@code mu} starts at 2.4 and keeps every candidate; accepting the selection moves {@code mu}
     * to 2.4 - 0.1 * (2.32193 - 1.2) = 2.28781, below every surprise, so the next row collapses to one candidate.
     * An apply that no accept follows adapts nothing.
     */
    @Test
    public void mirostatAdaptsMuOnTheAcceptedToken() {
        float[] uniform = new float[5];
        SamplerConfig config = plain().mirostat(2).mirostatTau(1.2f).build();
        Sampler accepted = sampler(config);
        int token = accepted.apply(uniform).selectedId();
        accepted.accept(token);
        assertEquals(1, accepted.apply(uniform).size());

        Sampler applied = sampler(config);
        applied.apply(uniform);
        assertEquals("an apply alone leaves mu", 5, applied.apply(uniform).size());

        Sampler outside = sampler(config);
        outside.apply(new float[]{0f, 0f, 0f, 0f, Float.NEGATIVE_INFINITY});
        outside.accept(4);
        assertEquals("a token outside the kept set leaves mu", 5, outside.apply(uniform).size());
    }

    /**
     * The speculative read-ahead (block, FLy): row 0 applied, the state saved, draft token 0 accepted, row 1 applied
     * under it and draft token 1 accepted; the draft is rejected at position 0, so the state is restored and the
     * replacement accepted. The chain then offers the same candidate sets as one that applied row 0 and accepted the
     * replacement alone.
     */
    @Test
    public void mirostatMuRestoresWithItsPendingRow() {
        for (int version = 1; version <= 2; version++) {
            SamplerConfig config = plain().mirostat(version).mirostatTau(3f).mirostatEta(1f).build();
            Sampler restored = sampler(config);
            Sampler reference = sampler(config);
            Sampler leaked = sampler(config);
            readAhead(restored, reference, leaked, zipf(7), certain(8));
            assertFalse("v" + version + ": the read-ahead must move mu for the check to mean anything",
                    Arrays.equals(candidateSet(leaked, zipf(9)), candidateSet(reference, zipf(9))));
            assertArrayEquals("v" + version, candidateSet(reference, zipf(9)), candidateSet(restored, zipf(9)), 0f);
        }
    }

    /**
     * Adaptive-p keeps its pending selection in the state: the restored chain folds the replacement into its average
     * as the reference does.
     */
    @Test
    public void adaptivePFoldsTheReplacementAfterARestore() {
        SamplerConfig config = plain().samplers("adaptive_p").adaptiveTarget(0.3f).adaptiveDecay(0.5f).build();
        Sampler restored = sampler(config);
        Sampler reference = sampler(config);
        Sampler unfolded = sampler(config);
        readAhead(restored, reference, unfolded, probe(7), probe(8));
        unfolded.reset();
        unfolded.apply(probe(7));
        assertFalse("folding the replacement must change the next row for the check to mean anything",
                Arrays.equals(candidateSet(unfolded, probe(9)), candidateSet(reference, probe(9))));
        assertArrayEquals(candidateSet(reference, probe(9)), candidateSet(restored, probe(9)), 0f);
    }

    /**
     * Runs the read-ahead protocol on {@code restored} (rows 0 and 1, draft tokens fed ahead, rolled back, the
     * replacement accepted), the plain one on {@code reference} (row 0, the replacement) and the read-ahead without
     * a rollback on {@code leaked}; the replacement is the token row 0 selected on {@code reference}.
     */
    private static void readAhead(Sampler restored, Sampler reference, Sampler leaked, float[] row0, float[] row1) {
        int replacement = reference.apply(row0).selectedId();
        reference.accept(replacement);
        int draft0 = (replacement + 1) % VOCAB.vocabSize();
        for (Sampler s : new Sampler[]{restored, leaked}) {
            s.apply(row0);
            Object state = s.saveState();
            s.accept(draft0);
            s.accept(s.apply(row1).selectedId());
            if (s == restored) {
                s.restoreState(state);
                s.accept(replacement);
            }
        }
    }
}

package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.Sampler;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.grammar.CompiledConstraint;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.gguf.grammar.GbnfBuilder;
import com.eonmux.neuralfree.gguf.grammar.Grammar;
import com.eonmux.neuralfree.gguf.grammar.GrammarTriggers;
import org.junit.Test;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * The trigger-gated grammar (llama.cpp's lazy grammar, xgrammar's tag dispatch): free text until a trigger fires,
 * the grammar from the trigger's start on with the overlapping tokens replayed, a LAZY grammar to the end and a
 * DISPATCH grammar back to free text after each completed structure; token, pattern and full-pattern triggers; the
 * pre-mask that keeps a token from completing a trigger word with text the grammar rejects; the free-text
 * excluded strings and tokens a structural tag hands the stage; prompt tokens and {@code reset} leave the stage
 * waiting.
 */
public class TriggeredGrammarStageTest {

    /** Byte tokens 1..127 render their byte, id 0 is EOG, and ids 128.. render the multi-byte {@link #EXTRA} pieces. */
    private static final String[] EXTRA = {"<t>", "<t>z", "ok\n<t>", "<call>"};

    private static final class Vocab implements SamplerVocab {
        @Override public int vocabSize() { return 128 + EXTRA.length; }
        @Override public boolean isEog(int id) { return id == 0; }
        @Override public int eotToken() { return 0; }
        @Override public int eosToken() { return 0; }
        @Override public byte[] piece(int id) {
            if (id <= 0) {
                return new byte[0];
            }
            return id < 128 ? new byte[]{(byte) id} : EXTRA[id - 128].getBytes(StandardCharsets.UTF_8);
        }
        @Override public String pieceText(int id) { return new String(piece(id), StandardCharsets.UTF_8); }
        @Override public int[] tokenize(String text) {
            return text.chars().toArray();
        }
    }

    private static final Vocab VOCAB = new Vocab();

    /** Greedy logits ranking {@code ids} in order (a repeated id keeps its first rank), every other id far below. */
    private static float[] prefer(int... ids) {
        float[] logits = new float[VOCAB.vocabSize()];
        Arrays.fill(logits, -100f);
        int rank = ids.length;
        for (int id : ids) {
            if (logits[id] == -100f) {
                logits[id] = rank;
            }
            rank--;
        }
        return logits;
    }

    /**
     * Greedy generation where the model "wants" the tokens of {@code script} in order (falling back to the EOG
     * token, then any token): the stage decides what survives.
     */
    private static String run(TriggeredGrammarStage stage, int[] script, int maxSteps) {
        Sampler sampler = Sampler.create(SamplerConfig.greedyDecoding(), new int[0], List.of(stage), VOCAB);
        StringBuilder out = new StringBuilder();
        for (int step = 0; step < maxSteps; step++) {
            int want = step < script.length ? script[step] : 0;
            int token = sampler.sample(prefer(want, 0, '}', '"', 'a', ':', '{', '1', '<', '/', 't', '>'));
            if (token == 0) {
                return out.toString();
            }
            out.append(VOCAB.pieceText(token));
            sampler.accept(token);
        }
        return out.toString();
    }

    private static int[] chars(String s) {
        return s.chars().toArray();
    }

    private static Grammar grammar(String gbnf) {
        return GbnfBuilder.grammarOf(ConstraintSpec.gbnf(gbnf));
    }

    @Test
    public void freeTextRunsUntilTheTriggerAndTheGrammarConstrainsFromItsStart() {
        Grammar g = grammar("root ::= \"<t>\" \"{\" \"\\\"a\\\"\" \":\" \"1\" \"}\" \"</t>\"");
        TriggeredGrammarStage stage = new TriggeredGrammarStage(g, GrammarTriggers.lazyWords(List.of("<t>")), VOCAB);
        // the model wants free text, then the trigger, then junk the grammar must override
        String out = run(stage, chars("hi <t>qqqqqqqqqqqqq"), 40);
        assertEquals("hi <t>{\"a\":1}</t>", out);
    }

    @Test
    public void aTriggerSplitAcrossTokensReplaysTheOverlappingPieces() {
        Grammar g = grammar("root ::= \"<t>x\"");
        TriggeredGrammarStage stage = new TriggeredGrammarStage(g, GrammarTriggers.lazyWords(List.of("<t>")), VOCAB);
        int[] script = {'a', 128 + 2 /* "ok\n<t>" */, 'q', 'q'};
        String out = run(stage, script, 10);
        assertEquals("aok\n<t>x", out);
    }

    @Test
    public void theLazyGrammarHoldsToTheEndOfTheGeneration() {
        Grammar g = grammar("root ::= \"<t>\" [a-z]");
        TriggeredGrammarStage stage = new TriggeredGrammarStage(g, GrammarTriggers.lazyWords(List.of("<t>")), VOCAB);
        // after the one letter the grammar is complete: only EOG remains, so the trailing free text never comes
        String out = run(stage, chars("<t>bmore text"), 20);
        assertEquals("<t>b", out);
    }

    @Test
    public void dispatchReturnsToFreeTextAfterEachStructure() {
        String payload = "{\"structures\": [{\"begin\": \"<call>\", \"schema\": {\"const\": 1}, \"end\": \"</call>\"}],"
                + " \"triggers\": [\"<call>\"]}";
        CompiledConstraint c = GbnfBuilder.compile(ConstraintSpec.structuralTag(payload));
        assertNotNull(c.triggers());
        assertEquals(GrammarTriggers.Mode.DISPATCH, c.triggers().mode());
        TriggeredGrammarStage stage = new TriggeredGrammarStage(c.grammar(), c.triggers(), VOCAB);
        String out = run(stage, chars("x <call>zzzzzzzzzzz then y <call>zzzzzzzzzzzzz"), 120);
        assertTrue("two constrained structures inside free text, got " + out,
                out.startsWith("x <call>1</call>") && out.contains("<call>1</call>"));
        assertEquals("the second trigger opens a second structure", 2, out.split("<call>1</call>", -1).length - 1);
    }

    @Test
    public void aTokenTriggerFiresOnTheSampledId() {
        Grammar g = grammar("root ::= \"<call>\" \"!\"");
        GrammarTriggers t = GrammarTriggers.builder(GrammarTriggers.Mode.LAZY).token(128 + 3).build();
        TriggeredGrammarStage stage = new TriggeredGrammarStage(g, t, VOCAB);
        String out = run(stage, new int[]{'h', 128 + 3, 'q', 'q'}, 10);
        assertEquals("h<call>!", out);
    }

    @Test
    public void aPatternTriggerStartsTheGrammarAtItsFirstGroup() {
        Grammar g = grammar("root ::= \"<t>\" \"y\"");
        GrammarTriggers t = GrammarTriggers.builder(GrammarTriggers.Mode.LAZY).pattern("\\s(<t>)").build();
        TriggeredGrammarStage stage = new TriggeredGrammarStage(g, t, VOCAB);
        String out = run(stage, chars("a<tb <t>qq"), 20);
        assertEquals("a<tb <t>y", out);
    }

    @Test
    public void aFullPatternMustMatchTheWholeOutput() {
        GrammarTriggers t = GrammarTriggers.builder(GrammarTriggers.Mode.LAZY).patternFull("[a-z]*(<t>)").build();
        assertEquals(2, t.firstStart("ab<t>"));
        assertEquals(-1, t.firstStart("a b<t>"));
    }

    @Test
    public void aTokenCompletingATriggerWithRejectedTextIsMasked() {
        Grammar g = grammar("root ::= \"<t>x\"");
        TriggeredGrammarStage stage = new TriggeredGrammarStage(g, GrammarTriggers.lazyWords(List.of("<t>")), VOCAB);
        // the model wants "<t>z" as one token: the grammar after <t> needs x, so that token is masked and the
        // plain "<t>" token is chosen instead
        Sampler sampler = Sampler.create(SamplerConfig.greedyDecoding(), new int[0], List.of(stage), VOCAB);
        int token = sampler.sample(prefer(128 + 1, 128));
        assertEquals("<t>", VOCAB.pieceText(token));
    }

    @Test
    public void freeTextExcludedStringsAreMaskedWhileWaiting() {
        Grammar g = grammar("root ::= \"<t>1\"");
        TriggeredGrammarStage stage = new TriggeredGrammarStage(g, GrammarTriggers.lazyWords(List.of("<t>")),
                VOCAB, List.of("no"), null);
        // the 'o' completing the excluded "no" is masked, so the model falls back to ending the output
        assertEquals("a n", run(stage, chars("a no"), 10));
        TriggeredGrammarStage okBanned = new TriggeredGrammarStage(g, GrammarTriggers.lazyWords(List.of("<t>")),
                VOCAB, List.of("ok"), null);
        // the piece "ok\n<t>" writes the excluded "ok" before its last byte: it is masked all the same
        assertEquals("", run(okBanned, new int[]{128 + 2}, 10));
    }

    @Test
    public void freeTextExcludedTokensAreMaskedWhileWaitingOnly() {
        Grammar g = grammar("root ::= \"<t>\" [q]");
        TriggeredGrammarStage stage = new TriggeredGrammarStage(g, GrammarTriggers.lazyWords(List.of("<t>")),
                VOCAB, null, new int[]{'q'});
        assertEquals("a", run(stage, chars("aq"), 10));
        stage.reset();
        // once the trigger fires the grammar alone decides, and it wants the excluded token
        assertEquals("<t>q", run(stage, chars("<t>q"), 10));
    }

    @Test
    public void promptTokensAndResetLeaveTheStageWaiting() {
        Grammar g = grammar("root ::= \"<t>x\"");
        TriggeredGrammarStage stage = new TriggeredGrammarStage(g, GrammarTriggers.lazyWords(List.of("<t>")), VOCAB);
        Sampler sampler = Sampler.create(SamplerConfig.greedyDecoding(), new int[0], List.of(stage), VOCAB);
        sampler.accept(chars("<t>"), 0, 3); // prompt history before the first apply
        assertTrue(stage.awaitingTrigger());
        assertEquals("free text is still free", 'q', sampler.sample(prefer('q')));
        sampler.accept('<');
        sampler.accept('t');
        sampler.accept('>');
        assertFalse(stage.awaitingTrigger());
        sampler.reset();
        assertTrue(stage.awaitingTrigger());
    }

    @Test
    public void anUngatedSpecCompilesWithoutTriggers() {
        CompiledConstraint c = GbnfBuilder.compile(ConstraintSpec.gbnf("root ::= \"a\""));
        assertNull(c.triggers());
        assertFalse(c.gated());
        CompiledConstraint gated = GbnfBuilder.compile(ConstraintSpec.gbnf("root ::= \"a\"")
                .withTriggers(GrammarTriggers.lazyWords(List.of("a"))));
        assertTrue(gated.gated());
    }
}

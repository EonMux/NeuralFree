package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.Sampler;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.gguf.grammar.GbnfBuilder;
import com.eonmux.neuralfree.gguf.grammar.Grammar;
import com.eonmux.neuralfree.gguf.grammar.GrammarTriggers;
import org.junit.Test;

import java.nio.charset.StandardCharsets;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * A trigger word whose multi-byte character is split across tokens: the waiting-phase check judges a candidate
 * that would complete the word the way the trigger replay will, token by token, so a token whose partial
 * character the grammar refuses where that token ends (a negated class overlapping the partial character's range,
 * llama.cpp's {@code match_partial_char}) is masked instead of being kept and then failing its replay.
 */
public class TriggerSplitCharacterTest {

    /** "é" is C3 A9; the first byte arrives alone, the second with the "x" that completes the trigger "éx". */
    private static final int LEAD = 128;
    private static final int TAIL_X = 129;

    private static final class Vocab implements SamplerVocab {
        @Override public int vocabSize() { return 130; }
        @Override public boolean isEog(int id) { return id == 0; }
        @Override public int eotToken() { return 0; }
        @Override public int eosToken() { return 0; }
        @Override public byte[] piece(int id) {
            if (id == LEAD) {
                return new byte[]{(byte) 0xC3};
            }
            if (id == TAIL_X) {
                return new byte[]{(byte) 0xA9, 'x'};
            }
            return id <= 0 ? new byte[0] : new byte[]{(byte) id};
        }
        @Override public String pieceText(int id) { return new String(piece(id), StandardCharsets.UTF_8); }
        @Override public int[] tokenize(String text) { return text.chars().toArray(); }
    }

    private static final Vocab VOCAB = new Vocab();

    private static TriggeredGrammarStage stage(String gbnf) {
        Grammar g = GbnfBuilder.grammarOf(ConstraintSpec.gbnf(gbnf));
        return new TriggeredGrammarStage(g, GrammarTriggers.lazyWords(List.of("éx")), VOCAB);
    }

    /** Arms the stage and buffers free text ending in the lone lead byte of "é". */
    private static void writeLeadByte(TriggeredGrammarStage stage) {
        stage.keeps('a');
        stage.accept('a');
        assertTrue(stage.keeps(LEAD));
        stage.accept(LEAD);
        assertTrue(stage.awaitingTrigger());
    }

    @Test
    public void aSplitCharacterTheReplayRefusesIsMasked() {
        // [^è] refuses the lead byte C3 on its own: è (U+00E8) lies in the range C3 can still complete to
        TriggeredGrammarStage stage = stage("root ::= [^è] \"x\" [a-z]*");
        writeLeadByte(stage);
        assertFalse("the replay of the lead byte fails, so completing the trigger is refused",
                stage.keeps(TAIL_X));
        assertFalse(stage.admits(TAIL_X));
        Sampler.Candidates candidates = Sampler.create(SamplerConfig.greedyDecoding(), new int[0], List.of(stage),
                VOCAB).apply(new float[VOCAB.vocabSize()]);
        assertEquals(Float.NEGATIVE_INFINITY, logitOf(candidates, TAIL_X), 0f);
        assertEquals(0f, logitOf(candidates, 'b'), 0f);
    }

    @Test
    public void aSplitCharacterTheReplayTakesStaysAvailable() {
        // [^z] takes the lead byte: nothing C3 can complete to is excluded
        TriggeredGrammarStage stage = stage("root ::= [^z] \"x\" [a-z]*");
        writeLeadByte(stage);
        assertTrue(stage.keeps(TAIL_X));
        stage.accept(TAIL_X);
        assertFalse("the trigger fired and the grammar took the replay", stage.awaitingTrigger());
        assertTrue(stage.keeps('q'));
        assertFalse(stage.keeps('7'));
    }

    private static float logitOf(Sampler.Candidates candidates, int id) {
        for (int i = 0; i < candidates.size(); i++) {
            if (candidates.id(i) == id) {
                return candidates.logit(i);
            }
        }
        throw new AssertionError("no candidate " + id);
    }
}

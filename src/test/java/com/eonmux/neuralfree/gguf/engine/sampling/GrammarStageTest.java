package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.Sampler;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.gguf.grammar.GbnfBuilder;
import com.eonmux.neuralfree.gguf.grammar.Grammar;
import com.eonmux.neuralfree.gguf.grammar.JsonSchemaToGbnf;
import org.junit.Test;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The grammar-constrained sampling stage (llama.cpp {@code llama-grammar.cpp} semantics, the guided decoding
 * route): the stage masks every token the grammar cannot accept next, the accepted tokens advance its pushdown
 * stacks through alternations, repetitions, character ranges and multi-byte UTF-8 pieces split across tokens, a
 * completed rule admits the end-of-generation tokens again, the prompt history before the first apply leaves the
 * initial stacks untouched and {@code reset} restores them.
 */
public class GrammarStageTest {

    /** ASCII byte tokens: id {@code c} renders the byte {@code c}; id 0 is EOG, ids 1..127 are their code point. */
    private static final class ByteVocab implements SamplerVocab {
        @Override public int vocabSize() { return 128; }
        @Override public boolean isEog(int id) { return id == 0; }
        @Override public int eotToken() { return 0; }
        @Override public int eosToken() { return 0; }
        @Override public byte[] piece(int id) {
            if (id <= 0) {
                return new byte[0];
            }
            return new byte[]{(byte) id};
        }
        @Override public String pieceText(int id) { return new String(piece(id), StandardCharsets.UTF_8); }
        @Override public int[] tokenize(String text) {
            byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
            int[] ids = new int[bytes.length];
            for (int i = 0; i < bytes.length; i++) {
                ids[i] = bytes[i] & 0xFF;
            }
            return ids;
        }
    }

    /** Extra UTF-8 tokens: id 128 = the two bytes of {@code é}, 129 = its lead byte alone, 130 = its tail alone. */
    private static final class Utf8Vocab implements SamplerVocab {
        private final ByteVocab base = new ByteVocab();

        @Override public int vocabSize() { return 131; }
        @Override public boolean isEog(int id) { return base.isEog(id); }
        @Override public int eotToken() { return base.eotToken(); }
        @Override public int eosToken() { return base.eosToken(); }
        @Override public byte[] piece(int id) {
            if (id == 128) {
                return new byte[]{(byte) 0xC3, (byte) 0xA9};
            }
            if (id == 129) {
                return new byte[]{(byte) 0xC3};
            }
            if (id == 130) {
                return new byte[]{(byte) 0xA9};
            }
            return base.piece(id);
        }
        @Override public String pieceText(int id) {
            return new String(piece(id), StandardCharsets.UTF_8);
        }
        @Override public int[] tokenize(String text) { return base.tokenize(text); }
    }

    /** Greedy logits favoring {@code ids} in order (every other candidate far below). */
    private static float[] prefer(int... ids) {
        float[] logits = new float[131];
        java.util.Arrays.fill(logits, -100f);
        int rank = ids.length;
        for (int id : ids) {
            logits[id] = rank--;
        }
        return logits;
    }

    /** Samples a whole generation: greedy chain with the stage, accepting every token into it. */
    private static String generate(Grammar grammar, SamplerVocab vocab, float[] logits) {
        Sampler sampler = Sampler.create(
                com.eonmux.neuralfree.gguf.engine.SamplerConfig.greedyDecoding(), new int[0],
                List.of(new GrammarStage(grammar, vocab)), vocab);
        StringBuilder out = new StringBuilder();
        for (int step = 0; step < 64; step++) {
            int token = sampler.sample(logits);
            if (token == 0) {
                return out.toString();
            }
            out.appendCodePoint(token);
            sampler.accept(token);
        }
        return out.toString();
    }

    /** A control token (id 120) whose plain piece is empty and whose special piece is {@code <|x|>}. */
    private static final class ControlVocab implements SamplerVocab {
        private final ByteVocab base = new ByteVocab();

        @Override public int vocabSize() { return 128; }
        @Override public boolean isEog(int id) { return base.isEog(id); }
        @Override public int eotToken() { return 0; }
        @Override public int eosToken() { return 0; }
        @Override public byte[] piece(int id) { return id == 120 ? new byte[0] : base.piece(id); }
        @Override public byte[] grammarPiece(int id) {
            return id == 120 ? "<|x|>".getBytes(StandardCharsets.UTF_8) : base.piece(id);
        }
        @Override public String pieceText(int id) { return new String(grammarPiece(id), StandardCharsets.UTF_8); }
        @Override public int[] tokenize(String text) { return base.tokenize(text); }
    }

    @Test
    public void aControlTokenMatchesByItsSpecialTextAndByItsId() {
        ControlVocab vocab = new ControlVocab();
        for (String rule : new String[]{"\"<|x|>\"", "<[120]>"}) {
            Grammar grammar = GbnfBuilder.grammarOf(ConstraintSpec.gbnf("root ::= " + rule + " \"a\""));
            Sampler sampler = Sampler.create(com.eonmux.neuralfree.gguf.engine.SamplerConfig.greedyDecoding(),
                    new int[0], List.of(new GrammarStage(grammar, vocab)), vocab);
            assertEquals(rule + " admits the control token first", 120, sampler.sample(prefer('a', 120)));
            sampler.accept(120);
            assertEquals(rule, 'a', sampler.sample(prefer(120, 'a')));
        }
    }

    @Test
    public void masksEveryTokenTheGrammarCannotAcceptAndForcesTheAlternation() {
        Grammar grammar = GbnfBuilder.grammarOf(ConstraintSpec.gbnf("root ::= \"yes\" | \"no\""));
        String out = generate(grammar, new ByteVocab(), prefer('x', 'y', 'n', 'o', 's', 'e', 0));
        assertTrue("the alternation forces yes or no, got " + out, out.equals("yes") || out.equals("no"));
    }

    @Test
    public void digitsThenANewlineCompleteTheRuleAndEogIsAdmittedAgain() {
        Grammar grammar = GbnfBuilder.grammarOf(ConstraintSpec.gbnf("root ::= [0-9]+ \"\\n\""));
        // the newline outranks the digit but masks at the rule start; after one digit it wins over the next digit
        String out = generate(grammar, new ByteVocab(), prefer('\n', '9', 'a', 0));
        assertEquals("9\n", out);
    }

    @Test
    public void promptTokensBeforeTheFirstApplyLeaveTheInitialStacks() {
        GrammarStage stage = new GrammarStage(
                GbnfBuilder.grammarOf(ConstraintSpec.gbnf("root ::= \"ok\"")), new ByteVocab());
        Sampler sampler = Sampler.create(
                com.eonmux.neuralfree.gguf.engine.SamplerConfig.greedyDecoding(), new int[0],
                List.of(stage), new ByteVocab());
        sampler.accept(new int[]{'x', 'y', 'z'}, 0, 3); // the prompt history
        int token = sampler.sample(prefer('o', 'k', 'x', 0));
        assertEquals("the unconsumed grammar still forces its first char", 'o', token);
        sampler.accept(token);
        token = sampler.sample(prefer('k', 'x', 0));
        assertEquals('k', token);
        sampler.accept(token);
        assertEquals("the rule completed: only the EOG token survives", 0, sampler.sample(prefer('x', 0)));
    }

    @Test
    public void aMultiByteCharacterSplitAcrossTokensAdvancesThroughThePartialTail() {
        Grammar grammar = GbnfBuilder.grammarOf(ConstraintSpec.gbnf("root ::= \"é\" \"!\""));
        GrammarStage stage = new GrammarStage(grammar, new Utf8Vocab());
        Sampler sampler = Sampler.create(
                com.eonmux.neuralfree.gguf.engine.SamplerConfig.greedyDecoding(), new int[0],
                List.of(stage), new Utf8Vocab());
        // after the lead byte (129) only a token whose piece STARTS with the continuation byte may follow —
        // at that point the whole-é token (128) restarts the character and masks (llama.cpp's decode_utf8
        // rejects a lead byte where a continuation is due); 130 wins here on rank anyway
        int token = sampler.sample(prefer('x', 129, 33, 0));
        assertEquals("the lead byte of é is the only viable start", 129, token);
        sampler.accept(token);
        token = sampler.sample(prefer('y', 33, 130, 0));
        assertEquals("the bare continuation byte completes é", 130, token);
        sampler.accept(token);
        token = sampler.sample(prefer('y', 33, 0));
        assertEquals("then ! follows", 33, token);
        sampler.accept(token);
        assertEquals("rule complete: EOG survives", 0, sampler.sample(prefer('z', 0)));
    }

    @Test
    public void everyAlternativeOfTheRootRuleStartsAParse() {
        // llama_grammar_init_impl: one initial stack per root alternative, an empty one completing at once
        Grammar grammar = GbnfBuilder.grammarOf(ConstraintSpec.gbnf("root ::= \"yes\" | \"no\" | \"maybe\""));
        assertEquals("no", generate(grammar, new ByteVocab(), prefer('n', 'o', 'y', 0)));
        assertEquals("maybe", generate(grammar, new ByteVocab(), prefer('m', 'a', 'y', 'b', 'e', 0)));
        Grammar optional = GbnfBuilder.grammarOf(ConstraintSpec.gbnf("root ::= | \"a\""));
        assertEquals("", generate(optional, new ByteVocab(), prefer(0, 'a')));
        assertEquals("a", generate(optional, new ByteVocab(), prefer('a', 0)));
        GrammarStage stage = new GrammarStage(grammar, new ByteVocab());
        Sampler sampler = Sampler.create(com.eonmux.neuralfree.gguf.engine.SamplerConfig.greedyDecoding(),
                new int[0], List.of(stage), new ByteVocab());
        sampler.accept(sampler.sample(prefer('y', 0)));
        stage.reset();
        assertEquals("reset restores every root alternative", 'm', sampler.sample(prefer('m', 'y', 0)));
    }

    @Test
    public void resetReturnsTheInitialStacks() {
        GrammarStage stage = new GrammarStage(
                GbnfBuilder.grammarOf(ConstraintSpec.gbnf("root ::= \"ab\"")), new ByteVocab());
        Sampler sampler = Sampler.create(
                com.eonmux.neuralfree.gguf.engine.SamplerConfig.greedyDecoding(), new int[0],
                List.of(stage), new ByteVocab());
        int token = sampler.sample(prefer('a', 'x', 0));
        assertEquals('a', token);
        sampler.accept(token);
        stage.reset();
        int again = sampler.sample(prefer('b', 'a', 'x', 0));
        assertEquals("back at the rule start: 'b' masks, 'a' forces", 'a', again);
    }

    @Test
    public void repetitionsAndRangesBoundTheRun() {
        Grammar grammar = GbnfBuilder.grammarOf(ConstraintSpec.gbnf("root ::= [a-c]{2}"));
        // exactly two chars of [a-c], then the rule completes: only EOG keeps a finite logit
        String out = generate(grammar, new ByteVocab(), prefer('c', 'a', 'b', 'z', 0));
        assertEquals(2, out.length());
        for (char c : out.toCharArray()) {
            assertTrue("only [a-c] survives the mask: " + out, c >= 'a' && c <= 'c');
        }
    }

    @Test
    public void aJsonSchemaGrammarShapesTheSampledText() {
        String schema = "{\"type\":\"object\",\"properties\":{\"ok\":{\"type\":\"boolean\"}},"
                + "\"required\":[\"ok\"],\"additionalProperties\":false}";
        Grammar grammar = GbnfBuilder.grammarOf(
                ConstraintSpec.jsonSchema(schema, JsonSchemaToGbnf.WhitespaceMode.FLEXIBLE));
        ByteVocab vocab = new ByteVocab();
        Sampler sampler = Sampler.create(
                com.eonmux.neuralfree.gguf.engine.SamplerConfig.greedyDecoding(), new int[0],
                List.of(new GrammarStage(grammar, vocab)), vocab);
        // constant preference order: the mask alone decides which bytes can come next
        StringBuilder out = new StringBuilder();
        float[] logits = new float[128];
        java.util.Arrays.fill(logits, -100f);
        logits['{'] = 10; // strong pull toward the object open; the mask reroutes when invalid
        logits['"'] = 9;
        logits['o'] = 8;
        logits['k'] = 7;
        logits[':'] = 6;
        logits['t'] = 5;
        logits['r'] = 4;
        logits['u'] = 3;
        logits['e'] = 2;
        logits['}'] = 1;
        for (int step = 0; step < 64; step++) {
            int token = sampler.sample(logits);
            if (token == 0) {
                break;
            }
            out.appendCodePoint(token);
            sampler.accept(token);
        }
        String json = out.toString();
        assertTrue("a complete object rendered: " + json, json.startsWith("{") && json.endsWith("}"));
        assertTrue("the required key appears: " + json, json.contains("\"ok\""));
        assertTrue("the boolean value appears: " + json, json.contains("true") || json.contains("false"));
    }

    @Test
    public void eogIsMaskedUntilARuleCompletes() {
        GrammarStage stage = new GrammarStage(
                GbnfBuilder.grammarOf(ConstraintSpec.gbnf("root ::= \"a\"")), new ByteVocab());
        Sampler sampler = Sampler.create(
                com.eonmux.neuralfree.gguf.engine.SamplerConfig.greedyDecoding(), new int[0],
                List.of(stage), new ByteVocab());
        float[] logits = prefer('a', 0);
        int token = sampler.sample(logits);
        assertEquals("mid-rule the EOG token masks", 'a', token);
        sampler.accept(token);
        assertEquals("rule complete: EOG survives", 0, sampler.sample(prefer('b', 0)));
    }

    @Test
    public void piecesRenderRoundTripThroughTheFakeVocab() {
        ByteVocab vocab = new ByteVocab();
        assertArrayEquals(new byte[]{'q'}, vocab.piece('q'));
        assertEquals(0, vocab.piece(0).length);
    }

    @Test
    public void theStageAlwaysAppliesUnderGreedyDecoding() {
        Grammar grammar = GbnfBuilder.grammarOf(ConstraintSpec.gbnf("root ::= \"m\""));
        GrammarStage stage = new GrammarStage(grammar, new ByteVocab());
        Sampler sampler = Sampler.create(
                com.eonmux.neuralfree.gguf.engine.SamplerConfig.greedyDecoding(), new int[0],
                List.of(stage), new ByteVocab());
        List<Integer> seen = new ArrayList<>();
        int token = sampler.sample(prefer('n', 'm', 0));
        seen.add(token);
        assertEquals("greedy argmax over the masked set picks 'm'", (Integer) (int) 'm', seen.get(0));
    }
}

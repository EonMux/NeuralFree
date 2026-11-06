package com.eonmux.neuralfree.gguf.grammar;

import com.eonmux.neuralfree.gguf.engine.Sampler;
import com.eonmux.neuralfree.gguf.engine.sampling.GrammarStage;
import com.eonmux.neuralfree.gguf.engine.sampling.SamplerVocab;
import org.junit.Test;

import java.nio.charset.StandardCharsets;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The structural-tag parser (PARITY SO-4): the vLLM legacy payload compiles to one grammar whose root
 * alternates the structures — each begin literal, the schema's grammar, the end literal — rules renaming under
 * a per-structure prefix so several structures compose; the compiled grammar constrains sampling through the
 * {@code GrammarStage}; the named refusals fire for a payload without structures, a structure no trigger opens,
 * or a structure missing a member. Trigger dispatch itself runs in {@code TriggeredGrammarStageTest}.
 */
public class StructuralTagTest {

    /** ASCII byte tokens: id {@code c} renders the byte {@code c}; id 0 is EOG; ids 128+ are unused. */
    private static final class ByteVocab implements SamplerVocab {
        @Override public int vocabSize() { return 128; }
        @Override public boolean isEog(int id) { return id == 0; }
        @Override public int eotToken() { return 0; }
        @Override public int eosToken() { return 0; }
        @Override public byte[] piece(int id) {
            return id <= 0 ? new byte[0] : new byte[]{(byte) id};
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

    private static float[] prefer(int... ids) {
        float[] out = new float[128];
        java.util.Arrays.fill(out, -100f);
        int rank = ids.length;
        for (int id : ids) {
            out[id] = rank--;
        }
        return out;
    }

    /** Greedy sampling with the compiled grammar until EOG (bounded). */
    private static String generate(Grammar grammar, String triggerPreference) {
        ByteVocab vocab = new ByteVocab();
        Sampler sampler = Sampler.create(
                com.eonmux.neuralfree.gguf.engine.SamplerConfig.greedyDecoding(), new int[0],
                java.util.List.of(new GrammarStage(grammar, vocab)), vocab);
        StringBuilder out = new StringBuilder();
        for (int step = 0; step < 400; step++) {
            int[] preferred = chars(triggerPreference + "qw");
            int token = sampler.sample(prefer(preferred));
            if (token == 0) {
                return out.toString();
            }
            out.appendCodePoint(token);
            sampler.accept(token);
        }
        return out.toString();
    }

    private static int[] chars(String s) {
        return s.chars().toArray();
    }

    @Test
    public void aStructureConstrainsTheOutputToBeginSchemaEnd() {
        String payload = "{\"structures\": [{\"begin\": \"<call>\","
                + "\"schema\": {\"type\": \"object\", \"properties\": {\"ok\": {\"const\": \"yes\"}},"
                + "\"required\": [\"ok\"], \"additionalProperties\": false},"
                + "\"end\": \"</call>\"}], \"triggers\": []}";
        Grammar grammar = GbnfBuilder.grammarOf(ConstraintSpec.structuralTag(payload));
        String out = generate(grammar, "<");
        assertTrue("the output is begin + schema + end, got " + out,
                out.startsWith("<call>") && out.endsWith("</call>") && out.contains("\"yes\""));
    }

    @Test
    public void twoStructuresAlternateUnderOneRoot() {
        String payload = "{\"structures\": ["
                + "{\"begin\": \"<a>\", \"schema\": {\"const\": \"one\"}, \"end\": \"</a>\"},"
                + "{\"begin\": \"<b>\", \"schema\": {\"const\": \"two\"}, \"end\": \"</b>\"}],"
                + "\"triggers\": []}";
        Grammar grammar = GbnfBuilder.grammarOf(ConstraintSpec.structuralTag(payload));
        String out = generate(grammar, "<b>");
        assertTrue("one of the two structures, got " + out,
                (out.startsWith("<a>") && out.endsWith("</a>") && out.contains("one"))
                        || (out.startsWith("<b>") && out.endsWith("</b>") && out.contains("two")));
    }

    @Test
    public void theNamedRefusalsFire() {
        assertRefused("{\"format\": {}}", "structures");
        assertRefused("{\"structures\": [{\"begin\": \"a\", \"schema\": {}, \"end\": \"b\"}],"
                + "\"triggers\": [\"<call>\"]}", "no trigger");
        assertRefused("{\"structures\": [{\"begin\": \"a\", \"schema\": {}, \"end\": \"b\"}],"
                + "\"triggers\": [\"\"]}", "trigger 0");
        assertRefused("{\"structures\": [{\"begin\": \"a\", \"end\": \"b\"}], \"triggers\": []}", "schema");
        assertRefused("{\"structures\": []}", "carries no structures");
    }

    private static void assertRefused(String payload, String messagePart) {
        try {
            GbnfBuilder.compile(ConstraintSpec.structuralTag(payload));
            throw new AssertionError("expected a refusal for " + payload);
        } catch (IllegalArgumentException e) {
            assertTrue("the refusal names the gap: " + e.getMessage(),
                    e.getMessage().contains(messagePart));
        }
    }
}

package com.eonmux.neuralfree.gguf.grammar;

import com.eonmux.neuralfree.gguf.engine.Sampler;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.sampling.GrammarStage;
import com.eonmux.neuralfree.gguf.engine.sampling.SamplerVocab;
import com.eonmux.neuralfree.gguf.engine.sampling.TriggeredGrammarStage;
import org.junit.Test;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The modern xgrammar structural-tag {@code format} object: the value formats ({@code const_string},
 * {@code json_schema}, {@code grammar}, {@code regex}, {@code token} and friends), the combinators
 * ({@code sequence}, {@code or}, {@code tag}, {@code optional}, {@code plus}, {@code star}, {@code repeat},
 * {@code tags_with_separator}), any text with and without excludes, the dispatching formats through the trigger
 * stage (string and token triggers, the forced single-tag plain grammar, the free-text excludes), the dispatching
 * formats nested in other formats and under {@code at_least_one} as plain grammars, type inference without a
 * {@code type} field, and the named refusals. The accept/reject verdicts of the nested and {@code at_least_one}
 * cases are xgrammar 0.2.1's for the same payloads, those of the earliest-trigger dispatch and the Lark grammar
 * xgrammar 0.2.8's.
 */
public class StructuralTagFormatTest {

    /** ASCII byte tokens: id {@code c} renders the byte {@code c}; id 0 is EOG. */
    private static final SamplerVocab VOCAB = new SamplerVocab() {
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
    };

    /** The vocabulary of a token-string test: the piece {@code <t>} is token 65, the piece {@code </t>} 66. */
    private static final GrammarVocab TOKEN_VOCAB = text -> {
        if ("<t>".equals(text)) {
            return new int[]{65};
        }
        if ("</t>".equals(text)) {
            return new int[]{66};
        }
        return new int[0];
    };

    /** A vocabulary whose ids 65 and 66 render the multi-byte pieces {@code <t>} and {@code </t>}. */
    private static final SamplerVocab PIECE_VOCAB = new SamplerVocab() {
        @Override public int vocabSize() { return 128; }
        @Override public boolean isEog(int id) { return id == 0; }
        @Override public int eotToken() { return 0; }
        @Override public int eosToken() { return 0; }
        @Override public byte[] piece(int id) {
            if (id == 65) {
                return "<t>".getBytes(StandardCharsets.UTF_8);
            }
            if (id == 66) {
                return "</t>".getBytes(StandardCharsets.UTF_8);
            }
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
    };

    private static final String CONST_ONE = "{\"type\": \"const_string\", \"value\": \"1\"}";

    private static Grammar grammar(String format) {
        return GbnfBuilder.grammarOf(ConstraintSpec.structuralTag("{\"format\": " + format + "}"));
    }

    private static CompiledConstraint compile(String format) {
        return GbnfBuilder.compile(ConstraintSpec.structuralTag("{\"format\": " + format + "}"));
    }

    private static CompiledConstraint compile(String format, GrammarVocab vocab) {
        return GbnfBuilder.compile(ConstraintSpec.structuralTag("{\"format\": " + format + "}"), vocab);
    }

    private static void assertMatches(String format, String text) {
        assertTrue("the format accepts " + text, GrammarMatcher.matches(grammar(format), text));
    }

    private static void assertRejects(String format, String text) {
        assertFalse("the format refuses " + text, GrammarMatcher.matches(grammar(format), text));
    }

    private static void assertRefused(String format, String messagePart) {
        try {
            grammar(format);
            throw new AssertionError("expected a refusal for " + format);
        } catch (IllegalArgumentException e) {
            assertTrue("the refusal names the gap: " + e.getMessage(), e.getMessage().contains(messagePart));
        }
    }

    // ------------------------------------------------------------------ value formats

    @Test
    public void constStringMatchesExactlyItsValue() {
        assertMatches("{\"type\": \"const_string\", \"value\": \"hello\"}", "hello");
        assertRejects("{\"type\": \"const_string\", \"value\": \"hello\"}", "hellx");
        assertMatches("{\"type\": \"const_string\", \"value\": \"\"}", "");
    }

    @Test
    public void sequenceOrdersItsElements() {
        String sequence = "{\"type\": \"sequence\", \"elements\": [" + CONST_ONE + ","
                + "{\"type\": \"const_string\", \"value\": \"x\"}]}";
        assertMatches(sequence, "1x");
        assertRejects(sequence, "x1");
    }

    @Test
    public void orAlternatesItsElements() {
        String or = "{\"type\": \"or\", \"elements\": ["
                + "{\"type\": \"const_string\", \"value\": \"cat\"},"
                + "{\"type\": \"const_string\", \"value\": \"dog\"}]}";
        assertMatches(or, "cat");
        assertMatches(or, "dog");
        assertRejects(or, "cow");
    }

    @Test
    public void regexCompilesItsPattern() {
        assertMatches("{\"type\": \"regex\", \"pattern\": \"[0-9]+\"}", "42");
        assertRejects("{\"type\": \"regex\", \"pattern\": \"[0-9]+\"}", "4x");
    }

    @Test
    public void regexMatchesWholeInTheXgrammarDialect() {
        String call = "{\"type\": \"regex\", \"pattern\": \"<call name=\\\"(?P<name>[\\\\w.-]+)\\\">\\\\n?\"}";
        assertMatches(call, "<call name=\"get.weather\">");
        assertMatches(call, "<call name=\"get.weather\">\n");
        assertRejects(call, "<call name=\"get weather\">");
        String negated = "{\"type\": \"regex\", \"pattern\": \"[^\\\\s<]+\"}";
        assertMatches(negated, "functions.bash");
        assertRejects(negated, "a b");
        assertRejects(negated, "a<b");
    }

    @Test
    public void regexDotMatchesLineBreaksAsXgrammarWritesIt() {
        assertMatches("{\"type\": \"regex\", \"pattern\": \"a.b\"}", "a\nb");
        assertMatches("{\"type\": \"regex\", \"pattern\": \"a.b\"}", "a\rb");
    }

    @Test
    public void grammarCompilesEbnf() {
        assertMatches("{\"type\": \"grammar\", \"grammar\": \"root ::= \\\"ab\\\"+\"}", "abab");
        assertRejects("{\"type\": \"grammar\", \"grammar\": \"root ::= \\\"ab\\\"+\"}", "abx");
    }

    @Test
    public void grammarCompilesLarkWhenTheSyntaxSaysSo() {
        String lark = "{\"type\": \"grammar\", \"syntax\": \"lark\", \"grammar\": \"start: \\\"ab\\\"+ N\\nN: /[0-9]/\"}";
        assertMatches(lark, "abab7");
        assertRejects(lark, "abab");
        String pairs = "{\"type\": \"grammar\", \"syntax\": \"lark\","
                + " \"grammar\": \"start: pair+\\npair: KEY \\\"=\\\" /[0-9]+/ \\\";\\\"\\nKEY: \\\"a\\\"..\\\"z\\\"\"}";
        assertMatches(pairs, "a=1;b=22;");
        assertRejects(pairs, "a=1");
        assertRejects(pairs, "A=1;");
        String inTag = "{\"type\": \"tag\", \"begin\": \"<x>\", \"end\": \"</x>\", \"content\": " + lark + "}";
        assertMatches(inTag, "<x>ab1</x>");
        assertRejects(inTag, "<x>1</x>");
    }

    @Test
    public void jsonSchemaTrueCompilesAnyJsonValue() {
        assertMatches("{\"type\": \"json_schema\", \"json_schema\": true}", "[1,2]");
        assertMatches("{\"type\": \"json_schema\", \"json_schema\": true}", "\"text\"");
    }

    @Test
    public void jsonSchemaObjectHoldsTheValueToTheSchema() {
        String schema = "{\"type\": \"json_schema\", \"json_schema\": {\"const\": \"yes\"}}";
        assertMatches(schema, "\"yes\"");
        assertRejects(schema, "\"no\"");
    }

    @Test
    public void jsonSchemaExcludesKeepTheStringsOutOfItsFreeStrings() {
        String f = "{\"type\": \"json_schema\", \"json_schema\": {\"type\": \"object\", \"properties\":"
                + " {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}, \"excludes\": [\"</call>\"]}";
        assertMatches(f, "{\"city\": \"Paris\"}");
        assertMatches(f, "{\"city\": \"a</cal>b\"}");
        assertRejects(f, "{\"city\": \"a</call>b\"}");
        assertRejects(f, "{\"town\": \"Paris\"}");
        String free = "{\"type\": \"json_schema\", \"json_schema\": true, \"excludes\": [\"</call>\"]}";
        assertMatches(free, "[1, {\"k\": \"v\"}]");
        assertRejects(free, "{\"</call>\": 1}");
        assertRejects(free, "\"\\\"</call>\"");
    }

    @Test
    public void jsonSchemaExcludesInsideATagHoldTheTagToItsEnd() {
        String f = "{\"type\": \"tag\", \"begin\": \"<call>\", \"content\": {\"type\": \"json_schema\","
                + " \"json_schema\": {\"type\": \"object\"}, \"excludes\": [\"</call>\"]}, \"end\": \"</call>\"}";
        assertMatches(f, "<call>{\"a\": \"b\"}</call>");
        assertRejects(f, "<call>{\"a\": \"</call>\"}</call>");
    }

    @Test
    public void jsonSchemaExcludesTheSchemaCannotHoldAreRefusedByName() {
        assertRefused("{\"type\": \"json_schema\", \"json_schema\": {\"type\": \"string\", \"maxLength\": 4},"
                + " \"excludes\": [\"x\"]}", "cannot hold the string excludes");
        assertRefused("{\"type\": \"json_schema\", \"json_schema\": {\"const\": \"a</c>\"}, \"excludes\": [\"</c>\"]}",
                "cannot hold the string excludes");
        assertRefused("{\"type\": \"json_schema\", \"json_schema\": {}, \"excludes\": [\"\"]}", "excludes[0]");
    }

    /**
     * The JSON schema coverage keywords under {@code excludes}: a format that may spell any text (email), key patterns
     * ({@code patternProperties}, {@code propertyNames}) and free keys beside declared ones are refused by name, since
     * their grammars have no excluding form; a date format beside an exclude it cannot spell still compiles.
     */
    @Test
    public void jsonSchemaCoverageKeywordsUnderExcludesAreRefusedByName() {
        String excludes = ", \"excludes\": [\"</c>\"]}";
        assertRefused("{\"type\": \"json_schema\", \"json_schema\": {\"type\": \"string\", \"format\": \"email\"}"
                + excludes, "a format string");
        assertRefused("{\"type\": \"json_schema\", \"json_schema\": {\"type\": \"object\", \"patternProperties\":"
                + " {\"^x-\": {\"type\": \"integer\"}}}" + excludes, "patternProperties keys");
        assertRefused("{\"type\": \"json_schema\", \"json_schema\": {\"type\": \"object\", \"propertyNames\":"
                + " {\"maxLength\": 3}}" + excludes, "cannot hold the string excludes");
        assertRefused("{\"type\": \"json_schema\", \"json_schema\": {\"type\": \"object\", \"properties\":"
                + " {\"a\": {\"type\": \"integer\"}}, \"additionalProperties\": true}" + excludes,
                "cannot hold the string excludes");
        assertMatches("{\"type\": \"json_schema\", \"json_schema\": {\"type\": \"string\", \"format\": \"date\"}"
                + excludes, "\"2024-01-31\"");
    }

    @Test
    public void tokenMatchesExactlyItsId() {
        assertMatches("{\"type\": \"token\", \"token\": 65}", "A");
        assertRejects("{\"type\": \"token\", \"token\": 65}", "B");
        assertRejects("{\"type\": \"token\", \"token\": 65}", "AA");
    }

    @Test
    public void excludeTokenMatchesOneOtherToken() {
        String exclude = "{\"type\": \"exclude_token\", \"exclude_tokens\": [65]}";
        assertMatches(exclude, "B");
        assertRejects(exclude, "A");
        assertRejects(exclude, "BB");
    }

    @Test
    public void excludeTokenRefusesEveryExcludedId() {
        String exclude = "{\"type\": \"exclude_token\", \"exclude_tokens\": [65, 66, 67]}";
        assertRejects(exclude, "A");
        assertRejects(exclude, "B");
        assertRejects(exclude, "C");
        assertMatches(exclude, "D");
    }

    @Test
    public void anyTokensMatchAnyRunButTheExcluded() {
        String any = "{\"type\": \"any_tokens\", \"exclude_tokens\": [65]}";
        assertMatches(any, "");
        assertMatches(any, "BCD");
        assertRejects(any, "A");
        assertRejects(any, "BAC");
    }

    // ------------------------------------------------------------------ combinators

    @Test
    public void tagEnforcesBeginContentEnd() {
        String tag = "{\"type\": \"tag\", \"begin\": \"<r>\","
                + " \"content\": {\"type\": \"json_schema\", \"json_schema\": {\"const\": \"yes\"}},"
                + " \"end\": \"</r>\"}";
        assertMatches(tag, "<r>\"yes\"</r>");
        assertRejects(tag, "<r>\"no\"</r>");
        assertRejects(tag, "<r>\"yes\"");
    }

    @Test
    public void tagAcceptsSeveralEndStrings() {
        String tag = "{\"type\": \"tag\", \"begin\": \"<r>\", \"content\": " + CONST_ONE + ","
                + " \"end\": [\"</a>\", \"</b>\"]}";
        assertMatches(tag, "<r>1</a>");
        assertMatches(tag, "<r>1</b>");
        assertRejects(tag, "<r>1</c>");
    }

    @Test
    public void tagsWithSeparatorRepeatTheTags() {
        String tags = "{\"type\": \"tags_with_separator\", \"separator\": \",\""
                + ", \"tags\": [" + tagOf("<a>", "1", "</a>") + ", " + tagOf("<b>", "2", "</b>") + "]}";
        assertMatches(tags, "");
        assertMatches(tags, "<a>1</a>");
        assertMatches(tags, "<a>1</a>,<b>2</b>");
        assertMatches(tags, "<b>2</b>,<a>1</a>,<b>2</b>");
        assertRejects(tags, "<a>1</a><b>2</b>");
        assertRejects(tags, "<a>1</a>,");
    }

    @Test
    public void tagsWithSeparatorAtLeastOneRequiresOne() {
        String tags = "{\"type\": \"tags_with_separator\", \"separator\": \",\", \"at_least_one\": true"
                + ", \"tags\": [" + tagOf("<a>", "1", "</a>") + "]}";
        assertMatches(tags, "<a>1</a>");
        assertRejects(tags, "");
    }

    @Test
    public void tagsWithSeparatorStopAfterFirstCapsTheRun() {
        String tags = "{\"type\": \"tags_with_separator\", \"separator\": \",\", \"stop_after_first\": true"
                + ", \"tags\": [" + tagOf("<a>", "1", "</a>") + "]}";
        assertMatches(tags, "<a>1</a>");
        assertRejects(tags, "<a>1</a>,<a>1</a>");
        assertMatches(tags, "");
    }

    @Test
    public void tagsWithSeparatorEmptySeparatorRunsThemBackToBack() {
        String tags = "{\"type\": \"tags_with_separator\", \"separator\": \"\""
                + ", \"tags\": [" + tagOf("<a>", "1", "</a>") + ", " + tagOf("<b>", "2", "</b>") + "]}";
        assertMatches(tags, "<a>1</a><b>2</b><a>1</a>");
        assertRejects(tags, "<a>1</a>,<b>2</b>");
    }

    @Test
    public void optionalPlusAndStarMatchTheirCounts() {
        String one = "{\"type\": \"const_string\", \"value\": \"ab\"}";
        assertMatches("{\"type\": \"optional\", \"content\": " + one + "}", "");
        assertMatches("{\"type\": \"optional\", \"content\": " + one + "}", "ab");
        assertRejects("{\"type\": \"optional\", \"content\": " + one + "}", "abab");
        assertMatches("{\"type\": \"plus\", \"content\": " + one + "}", "abab");
        assertRejects("{\"type\": \"plus\", \"content\": " + one + "}", "");
        assertMatches("{\"type\": \"star\", \"content\": " + one + "}", "");
        assertMatches("{\"type\": \"star\", \"content\": " + one + "}", "abab");
        assertRejects("{\"type\": \"star\", \"content\": " + one + "}", "aba");
    }

    @Test
    public void repeatMatchesItsBounds() {
        String repeat = "{\"type\": \"repeat\", \"min\": 2, \"max\": 3, \"content\": "
                + "{\"type\": \"const_string\", \"value\": \"a\"}}";
        assertRejects(repeat, "");
        assertRejects(repeat, "a");
        assertMatches(repeat, "aa");
        assertMatches(repeat, "aaa");
        assertRejects(repeat, "aaaa");
    }

    @Test
    public void repeatUnboundedMeansMinOrMore() {
        String repeat = "{\"type\": \"repeat\", \"min\": 1, \"max\": -1, \"content\": "
                + "{\"type\": \"const_string\", \"value\": \"a\"}}";
        assertRejects(repeat, "");
        assertMatches(repeat, "aaaa");
    }

    @Test
    public void starOverAnyTextMatchesAnyText() {
        assertMatches("{\"type\": \"star\", \"content\": {\"type\": \"any_text\"}}", "");
        assertMatches("{\"type\": \"star\", \"content\": {\"type\": \"any_text\"}}", "anything goes");
    }

    @Test
    public void anyTextWithoutExcludesMatchesAnything() {
        assertMatches("{\"type\": \"any_text\"}", "");
        assertMatches("{\"type\": \"any_text\"}", "anything, <tags> and \"quotes\"");
    }

    @Test
    public void anyTextWithExcludesNeverWritesThem() {
        String any = "{\"type\": \"any_text\", \"excludes\": [\"nope\", \"never\"]}";
        assertMatches(any, "free text with other words");
        assertRejects(any, "this is nope at all");
        assertRejects(any, "it is never done");
    }

    @Test
    public void anyTextInsideATagKeepsTheEndStringReachable() {
        String tag = "{\"type\": \"tag\", \"begin\": \"<t>\", \"content\": {\"type\": \"any_text\"},"
                + " \"end\": \"</t>\"}";
        assertMatches(tag, "<t>free text</t>");
        assertRejects(tag, "<t>swallowed </t> and more</t>");
    }

    // ------------------------------------------------------------------ inference

    @Test
    public void aTagShapeInfersWithoutAType() {
        assertMatches("{\"begin\": \"<x>\", \"content\": " + CONST_ONE + ", \"end\": \"</x>\"}", "<x>1</x>");
        assertRejects("{\"begin\": \"<x>\", \"content\": " + CONST_ONE + ", \"end\": \"</x>\"}", "<y>1</x>");
    }

    @Test
    public void aConstantShapeInfersWithoutAType() {
        assertMatches("{\"value\": \"z\"}", "z");
    }

    @Test
    public void anEmptyObjectInfersNothing() {
        assertRefused("{}", "matches no supported type");
    }

    // ------------------------------------------------------------------ dispatching formats

    @Test
    public void triggeredTagsDispatchFreeTextThroughTheTrigger() {
        CompiledConstraint compiled = compile("{\"type\": \"triggered_tags\", \"triggers\": [\"<call\"],"
                + " \"tags\": [" + tagOf("<call>", "1", "</call>") + "]}");
        assertTrue(compiled.gated());
        assertEquals(GrammarTriggers.Mode.DISPATCH, compiled.triggers().mode());
        String out = run(compiled, "x <call>zzzzzzzzz then y <call>zzzzzzzzzz", 60);
        assertTrue("two dispatched calls inside free text, got " + out,
                out.startsWith("x <call>1</call>") && out.split("<call>1</call>", -1).length - 1 == 2);
    }

    @Test
    public void triggeredTagsStopAfterFirstHoldTheGrammarToTheEnd() {
        CompiledConstraint compiled = compile("{\"type\": \"triggered_tags\", \"triggers\": [\"<call\"],"
                + " \"stop_after_first\": true, \"tags\": [" + tagOf("<call>", "1", "</call>") + "]}");
        assertEquals(GrammarTriggers.Mode.LAZY, compiled.triggers().mode());
        String out = run(compiled, "<call>zzzzzz more text", 30);
        assertEquals("<call>1</call>", out);
    }

    @Test
    public void triggeredTagsForcedOneIsAPlainGrammar() {
        CompiledConstraint compiled = compile("{\"type\": \"triggered_tags\", \"triggers\": [\"<call\"],"
                + " \"at_least_one\": true, \"stop_after_first\": true,"
                + " \"tags\": [" + tagOf("<call>", "1", "</call>") + "]}");
        assertFalse("the forced single tag needs no trigger", compiled.gated());
        assertTrue(GrammarMatcher.matches(compiled.grammar(), "<call>1</call>"));
        assertFalse(GrammarMatcher.matches(compiled.grammar(), ""));
    }

    @Test
    public void triggeredTagsShareOneTriggerAcrossTags() {
        CompiledConstraint compiled = compile("{\"type\": \"triggered_tags\", \"triggers\": [\"<c\"],"
                + " \"tags\": [" + tagOf("<call>", "1", "</call>") + ", " + tagOf("<cash>", "2", "</cash>") + "]}");
        String out = run(compiled, "x <cash>zzzz", 30);
        assertTrue("the trigger opens either tag, got " + out, out.startsWith("x <cash>2</cash>"));
    }

    @Test
    public void triggeredTagsExcludesBanTheStringsFromTheFreeText() {
        CompiledConstraint compiled = compile("{\"type\": \"triggered_tags\", \"triggers\": [\"<call\"],"
                + " \"excludes\": [\"<call\"], \"tags\": [" + tagOf("<call>", "1", "</call>") + "]}");
        String out = run(compiled, "hi <call>", 12);
        assertFalse("the free text never spells the excluded string, got " + out,
                out.contains("<call"));
    }

    @Test
    public void dispatchRunsItsFormatAfterThePattern() {
        CompiledConstraint looping = compile("{\"type\": \"dispatch\","
                + " \"rules\": [[\"<q>\", " + CONST_ONE + "]]}");
        assertEquals(GrammarTriggers.Mode.DISPATCH, looping.triggers().mode());
        String out = run(looping, "x <q>zzz <q>z <q>zzzzz", 40);
        assertTrue("three dispatched runs, got " + out,
                out.startsWith("x <q>1") && out.split("<q>1", -1).length - 1 == 3);

        CompiledConstraint once = compile("{\"type\": \"dispatch\", \"loop\": false,"
                + " \"rules\": [[\"<q>\", " + CONST_ONE + "]]}");
        assertEquals(GrammarTriggers.Mode.LAZY, once.triggers().mode());
        assertEquals("x <q>1", run(once, "x <q>zzzzzzz more", 40));
    }

    @Test
    public void tokenTriggeredTagsDispatchOnTheSampledToken() {
        CompiledConstraint compiled = compile("{\"type\": \"token_triggered_tags\", \"trigger_tokens\": [60],"
                + " \"tags\": [{\"begin\": {\"type\": \"token\", \"token\": 60}, \"content\": "
                + "{\"type\": \"const_string\", \"value\": \"hi\"}, \"end\": \"</c>\"}]}");
        assertTrue(compiled.gated());
        assertEquals("a<hi</c>", run(compiled, "a<zz", 30));
    }

    @Test
    public void tokenTriggeredTagsForcedOneIsAPlainGrammar() {
        CompiledConstraint compiled = compile("{\"type\": \"token_triggered_tags\", \"trigger_tokens\": [60],"
                + " \"at_least_one\": true, \"stop_after_first\": true,"
                + " \"tags\": [{\"begin\": {\"type\": \"token\", \"token\": 60}, \"content\": "
                + "{\"type\": \"const_string\", \"value\": \"hi\"}, \"end\": \"</c>\"}]}");
        assertFalse(compiled.gated());
        assertTrue(GrammarMatcher.matches(compiled.grammar(), "<hi</c>"));
    }

    @Test
    public void tokenTriggeredTagsResolveTokenStringsThroughTheVocabulary() {
        CompiledConstraint compiled = compile("{\"type\": \"token_triggered_tags\","
                + " \"trigger_tokens\": [\"<t>\"],"
                + " \"tags\": [{\"begin\": {\"type\": \"token\", \"token\": \"<t>\"}, \"content\": "
                + "{\"type\": \"const_string\", \"value\": \"ok\"}, \"end\": {\"type\": \"token\","
                + " \"token\": \"</t>\"}}]}", TOKEN_VOCAB);
        assertTrue(compiled.gated());
        assertEquals("a<t>ok</t>", run(compiled, new int[]{'a', 65, 'z', 'z'}, 30, PIECE_VOCAB));
    }

    @Test
    public void tokenTriggeredTagsRefuseTokenStringsWithoutTheVocabulary() {
        assertRefused("{\"type\": \"token_triggered_tags\", \"trigger_tokens\": [\"<t>\"],"
                + " \"tags\": [{\"begin\": {\"type\": \"token\", \"token\": \"<t>\"}, \"content\": "
                + CONST_ONE + ", \"end\": \"</c>\"}]}", "needs the model vocabulary");
    }

    @Test
    public void tokenDispatchRunsItsFormatAfterTheToken() {
        CompiledConstraint compiled = compile("{\"type\": \"token_dispatch\","
                + " \"rules\": [[60, " + CONST_ONE + "]]}");
        assertEquals("a<1z", run(compiled, "a<zz", 20));
    }

    @Test
    public void anInferredDispatchingShapeCompilesAtTheTopLevel() {
        CompiledConstraint compiled = compile("{\"triggers\": [\"<call\"], \"tags\": ["
                + tagOf("<call>", "1", "</call>") + "]}");
        assertTrue(compiled.gated());
        assertEquals(GrammarTriggers.Mode.DISPATCH, compiled.triggers().mode());
    }

    // ------------------------------------------------------------------ nested dispatching formats

    /** xgrammar's builtin tool-calling shape: a reasoning prefix up to {@code </think>}, then triggered tool calls. */
    private static final String REASONING_THEN_CALLS = "{\"type\": \"sequence\", \"elements\": ["
            + "{\"type\": \"tag\", \"begin\": \"\", \"content\": {\"type\": \"any_text\"}, \"end\": \"</think>\"},"
            + "{\"type\": \"triggered_tags\", \"triggers\": [\"<call>\"], \"excludes\": [\"<think>\"],"
            + " \"tags\": [" + tagOf("<call>", "1", "</call>") + "]}]}";

    @Test
    public void triggeredTagsNestedInASequenceCompileToAPlainGrammar() {
        assertFalse("a nested dispatch runs in the grammar itself", compile(REASONING_THEN_CALLS).gated());
        assertMatches(REASONING_THEN_CALLS, "</think>");
        assertMatches(REASONING_THEN_CALLS, "abc</think>");
        assertMatches(REASONING_THEN_CALLS, "abc</think>hi <call>1</call> bye");
        assertMatches(REASONING_THEN_CALLS, "abc</think>x</think>");
        assertRejects(REASONING_THEN_CALLS, "abc</think>hi <call>2</call>");
        assertRejects(REASONING_THEN_CALLS, "abc</think>hi <think>");
        assertRejects(REASONING_THEN_CALLS, "abc");
    }

    @Test
    public void triggeredTagsInsideATagNeverWriteTheTagEnd() {
        String tag = "{\"type\": \"tag\", \"begin\": \"<a>\", \"content\": {\"type\": \"triggered_tags\","
                + " \"triggers\": [\"<c\"], \"tags\": [" + tagOf("<call>", "1", "</call>") + "]}, \"end\": \"</a>\"}";
        assertMatches(tag, "<a></a>");
        assertMatches(tag, "<a>x<call>1</call></a>");
        assertRejects(tag, "<a>x</a>y</a>");
        assertRejects(tag, "<a>x<c</a>");
    }

    @Test
    public void stopAfterFirstNestedEndsTheFormatAfterOneTag() {
        String sequence = "{\"type\": \"sequence\", \"elements\": [{\"type\": \"const_string\", \"value\": \"S\"},"
                + "{\"type\": \"triggered_tags\", \"triggers\": [\"<c\"], \"stop_after_first\": true, \"tags\": ["
                + tagOf("<call>", "1", "</call>") + "]}, {\"type\": \"const_string\", \"value\": \"E\"}]}";
        assertMatches(sequence, "SE");
        assertMatches(sequence, "SxE");
        assertMatches(sequence, "Sx<call>1</call>E");
        assertRejects(sequence, "Sx<call>1</call>yE");
        assertRejects(sequence, "S<call>1</call><call>1</call>E");
    }

    @Test
    public void atLeastOneOpensWithATagThenDispatches() {
        String top = "{\"type\": \"triggered_tags\", \"triggers\": [\"<c\"], \"at_least_one\": true, \"tags\": ["
                + tagOf("<call>", "1", "</call>") + "]}";
        assertFalse("the forced opening tag is a plain grammar", compile(top).gated());
        assertRejects(top, "");
        assertRejects(top, "x<call>1</call>");
        assertMatches(top, "<call>1</call>abc");
        assertMatches(top, "<call>1</call>x<call>1</call>yy");

        String nested = "{\"type\": \"sequence\", \"elements\": [{\"type\": \"const_string\", \"value\": \"S\"}, "
                + top + "]}";
        assertRejects(nested, "S");
        assertMatches(nested, "S<call>1</call>");
        assertRejects(nested, "Sx<call>1</call>");
    }

    @Test
    public void nestedDispatchFiresAtTheEarliestTriggerOccurrence() {
        String overlapping = "{\"type\": \"sequence\", \"elements\": [{\"type\": \"const_string\", \"value\": \"S\"},"
                + "{\"type\": \"triggered_tags\", \"triggers\": [\"aba\"], \"tags\": ["
                + tagOf("aba", "X", "!") + "]}]}";
        assertMatches(overlapping, "SabaX!");
        assertMatches(overlapping, "SabX!");
        assertMatches(overlapping, "Sab");
        assertMatches(overlapping, "SabaX!abaX!");
        assertRejects(overlapping, "SababaX!");
        assertRejects(overlapping, "SabaX!ababaX!");

        String crossing = "{\"type\": \"sequence\", \"elements\": [{\"type\": \"const_string\", \"value\": \"S\"},"
                + "{\"type\": \"triggered_tags\", \"triggers\": [\"<a\", \"a<\"], \"tags\": ["
                + tagOf("<a>", "X", "!") + "," + tagOf("a<b>", "Y", "!") + "]}]}";
        assertMatches(crossing, "S<a>X!");
        assertMatches(crossing, "Sa<b>Y!");
        assertRejects(crossing, "Sa<a>X!");
        assertRejects(crossing, "Sxa<a>X!");

        String once = "{\"type\": \"sequence\", \"elements\": [{\"type\": \"const_string\", \"value\": \"S\"},"
                + "{\"type\": \"dispatch\", \"rules\": [[\"aa\", {\"type\": \"const_string\", \"value\": \"X\"}]],"
                + " \"loop\": false}]}";
        assertMatches(once, "SaaX");
        assertMatches(once, "SaX");
        assertRejects(once, "SaaaX");
    }

    @Test
    public void dispatchNestedRunsItsRulesUntilTheFollowingFormat() {
        String looping = "{\"type\": \"sequence\", \"elements\": [{\"type\": \"dispatch\", \"rules\": [[\"<q>\", "
                + CONST_ONE + "]], \"excludes\": [\"<e>\"]}, {\"type\": \"const_string\", \"value\": \"<e>\"}]}";
        assertMatches(looping, "<e>");
        assertMatches(looping, "ab<q>1cd<q>1<e>");
        assertRejects(looping, "ab<q>2<e>");
        assertRejects(looping, "a<e>b<e>");

        String once = "{\"type\": \"tag\", \"begin\": \"<a>\", \"content\": {\"type\": \"dispatch\", \"rules\": [[\"<q>\", "
                + CONST_ONE + "]], \"loop\": false, \"excludes\": [\"<e>\"]}, \"end\": \"</a>\"}";
        assertMatches(once, "<a>xx</a>");
        assertMatches(once, "<a>x<q>1</a>");
        assertRejects(once, "<a>x<q>1y</a>");
        assertRejects(once, "<a>x<q>1<q>1</a>");
        assertRejects(once, "<a>x<e></a>");
    }

    @Test
    public void tokenTriggeredTagsNestedExcludeTheirTokensAndTheEndToken() {
        String tag = "{\"type\": \"tag\", \"begin\": \"[\", \"content\": {\"type\": \"token_triggered_tags\","
                + " \"trigger_tokens\": [60], \"exclude_tokens\": [120], \"tags\": [{\"begin\": {\"type\": \"token\","
                + " \"token\": 60}, \"content\": {\"type\": \"const_string\", \"value\": \"hi\"}, \"end\": \">\"}]},"
                + " \"end\": {\"type\": \"token\", \"token\": 93}}";
        assertMatches(tag, "[]");
        assertMatches(tag, "[ab<hi>c]");
        assertRejects(tag, "[axb]");
        assertRejects(tag, "[a<ho>]");
        assertRejects(tag, "[a]b]");
    }

    @Test
    public void tokenDispatchNestedRunsItsRuleAfterTheToken() {
        String sequence = "{\"type\": \"sequence\", \"elements\": [{\"type\": \"token_dispatch\", \"loop\": false,"
                + " \"rules\": [[60, " + CONST_ONE + "]], \"exclude_tokens\": [69]}, {\"type\": \"token\", \"token\": 69}]}";
        assertMatches(sequence, "abE");
        assertMatches(sequence, "a<1E");
        assertRejects(sequence, "a<2E");
        assertRejects(sequence, "a<1b<1E");
    }

    @Test
    public void aDispatchingShapeInfersInsideAnotherFormat() {
        String sequence = "{\"type\": \"sequence\", \"elements\": [{\"triggers\": [\"<c\"], \"tags\": ["
                + tagOf("<call>", "1", "</call>") + "]}]}";
        assertMatches(sequence, "x<call>1</call>y");
        assertRejects(sequence, "x<call>2</call>");
    }

    // ------------------------------------------------------------------ refusals

    @Test
    public void theNamedRefusalsFire() {
        assertRefused("{\"type\": \"nope\"}", "not recognized");
        assertRefused("{\"type\": \"sequence\", \"elements\": []}", "at least one element");
        assertRefused("{\"type\": \"or\", \"elements\": [\"x\"]}", "must be an object");
        assertRefused("{\"type\": \"grammar\", \"grammar\": \"root ::= \\\"a\\\"\", \"syntax\": \"peg\"}",
                "syntax 'peg'");
        assertRefused("{\"type\": \"grammar\", \"grammar\": \"root ::= \\\"a\\\"\", \"syntax\": \"lark\"}",
                "'::=' is GBNF rule syntax");
        assertRefused("{\"type\": \"any_text\", \"excludes\": [\"\"]}", "excludes 0");
        assertRefused("{\"type\": \"any_text\", \"max_chars\": -1}",
                "max_chars must be an integer from 0 to 2147483647, got -1");
        assertRefused("{\"type\": \"any_tokens\", \"max_tokens\": \"2\"}", "max_tokens must be an integer");
        assertRefused("{\"type\": \"repeat\", \"min\": 3, \"max\": 2, \"content\": " + CONST_ONE + "}",
                "min 3 is above max 2");
        assertRefused("{\"type\": \"repeat\", \"min\": -1, \"max\": 2, \"content\": " + CONST_ONE + "}",
                "min must be >= 0");
        assertRefused("{\"type\": \"tag\", \"begin\": \"<\", \"content\": {\"type\": \"any_text\"},"
                + " \"end\": \"\"}", "at least one end string must be non-empty");
        assertRefused("{\"type\": \"tag\", \"begin\": \"<\", \"content\": " + CONST_ONE + "}",
                "end field");
        assertRefused("{\"type\": \"tag\", \"begin\": 5, \"content\": " + CONST_ONE + ", \"end\": \">\"}",
                "begin must be a string or a token object");
        assertRefused("{\"type\": \"tags_with_separator\", \"separator\": \",\", \"tags\": []}",
                "at least one tag");
        assertRefused("{\"type\": \"tags_with_separator\", \"tags\": [" + tagOf("<a>", "1", "</a>") + "]}",
                "separator field");
        assertRefused("{\"type\": \"triggered_tags\", \"triggers\": [], \"tags\": ["
                + tagOf("<call>", "1", "</call>") + "]}", "at least one trigger");
        assertRefused("{\"type\": \"triggered_tags\", \"triggers\": [\"<call\"], \"tags\": []}",
                "at least one tag");
        assertRefused("{\"type\": \"triggered_tags\", \"triggers\": [\"x\"], \"tags\": ["
                + tagOf("<call>", "1", "</call>") + "]}", "which no trigger");
        assertRefused("{\"type\": \"triggered_tags\", \"triggers\": [\"<\", \"<c\"], \"tags\": ["
                + tagOf("<call>", "1", "</call>") + "]}", "several triggers");
        assertRefused("{\"type\": \"triggered_tags\", \"triggers\": [\"<call\"], \"tags\": [{\"begin\":"
                + " {\"type\": \"token\", \"token\": 60}, \"content\": " + CONST_ONE + ", \"end\": \"x\"}]}",
                "needs a string begin");
        assertRefused("{\"type\": \"dispatch\", \"rules\": []}", "at least one rule");
        assertRefused("{\"type\": \"dispatch\", \"rules\": [[\"\", " + CONST_ONE + "]]}",
                "non-empty pattern string");
        assertRefused("{\"type\": \"dispatch\", \"rules\": [[\"<q>\"]]}", "two-element array");
        assertRefused("{\"type\": \"token_dispatch\", \"rules\": [[-1, " + CONST_ONE + "]]}",
                "non-negative token id");
    }

    // ------------------------------------------------------------------ harness

    private static String tagOf(String begin, String value, String end) {
        return "{\"begin\": \"" + begin + "\", \"content\": {\"type\": \"const_string\", \"value\": \""
                + value + "\"}, \"end\": \"" + end + "\"}";
    }

    /** Greedy logits ranking {@code ids} in order, every other id far below. */
    private static float[] prefer(SamplerVocab vocab, int... ids) {
        float[] logits = new float[vocab.vocabSize()];
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
     * Greedy generation where the model "wants" the characters of {@code script} in order (then the EOG token and
     * a junk fallback the grammar must override): the compiled constraint decides what survives.
     */
    private static String run(CompiledConstraint compiled, String script, int maxSteps) {
        return run(compiled, script.chars().toArray(), maxSteps, VOCAB);
    }

    private static String run(CompiledConstraint compiled, int[] script, int maxSteps, SamplerVocab vocab) {
        Sampler sampler = Sampler.create(SamplerConfig.greedyDecoding(), new int[0],
                List.of(new TriggeredGrammarStage(compiled.grammar(), compiled.triggers(), vocab,
                        compiled.freeExcludes(), compiled.freeExcludeTokens())), vocab);
        StringBuilder out = new StringBuilder();
        for (int step = 0; step < maxSteps; step++) {
            int next = step < script.length ? script[step] : 0;
            int token = sampler.sample(prefer(vocab, next, 0, 'z', '"', '}', ':', '1', '<', '/', 'c', 'a', 's',
                    'l', '>', 't', 'h', 'i', 'o', 'k', 'b'));
            if (token == 0) {
                return out.toString();
            }
            out.append(vocab.pieceText(token));
            sampler.accept(token);
        }
        return out.toString();
    }
}

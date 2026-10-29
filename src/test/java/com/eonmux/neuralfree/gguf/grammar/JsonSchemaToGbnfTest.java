package com.eonmux.neuralfree.gguf.grammar;

import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Tests of the JSON schema to GBNF translator ({@link JsonSchemaToGbnf}), ported from llama.cpp's
 * {@code tests/test-json-schema-to-grammar.cpp} golden cases: every expected grammar string is the
 * C++ test's expected grammar, translated to the exact output of this port (rule registry formatted
 * in name order, llama.cpp's {@code format_grammar}). The C++ case names are cited per test. Also
 * the alpha WP clause groups: the compact whitespace mode of row G8, the {@code choice} alternation
 * of {@code ConstraintSpec}, and the refusals (remote {@code $ref}, unsupported and invalid
 * regexps, Lark {@code %}-imports). Pure logic; no model file required.
 */
public class JsonSchemaToGbnfTest {

    /** llama.cpp {@code SPACE_RULE}, the flexible whitespace mode (row G8). */
    private static final String SPACE = "| \" \" | \"\\n\"{1,2} [ \\t]{0,20}";

    /** llama.cpp {@code PRIMITIVE_RULES} fragments, verbatim {@code common/json-schema-to-grammar.cpp:236-258}. */
    private static final String CHAR_RULE =
            "[^\"\\\\\\x7F\\x00-\\x1F] | [\\\\] ([\"\\\\bfnrt] | \"u\" [0-9a-fA-F]{4})";
    private static final String STRING_RULE = "\"\\\"\" char* \"\\\"\"";
    private static final String NUMBER_RULE =
            "(\"-\"? integral-part) (\".\" decimal-part)? ([eE] [-+]? integral-part)?";
    private static final String DECIMAL_PART = "[0-9]{1,16}";
    private static final String INTEGRAL_PART = "[0] | [1-9] [0-9]{0,15}";

    private static String line(String name, String rule) {
        return name + " ::= " + rule;
    }

    private static String lines(String... parts) {
        return String.join("\n", parts);
    }

    /** The trimmed schema JSON of one C++ test case, converted in the C++ default mode. */
    private static String flex(String schema) {
        return JsonSchemaToGbnf.gbnf(schema, JsonSchemaToGbnf.WhitespaceMode.FLEXIBLE);
    }

    private static String spaceLine() {
        return line("space", SPACE);
    }

    // -------------------------------------------------- LC test "min 0 max 23" (:229-240)

    @Test
    public void integerMinMax24HourClock() {
        assertEquals(lines(
                line("root", "([0-9] | ([1] [0-9] | [2] [0-3]))"),
                spaceLine()),
                flex("{\"type\": \"integer\", \"minimum\": 0, \"maximum\": 23}"));
    }

    // -------------------------------------------------- LC test "min 15 max 300" (:242-254)

    @Test
    public void integerMinMaxAcrossDigitLengths() {
        assertEquals(lines(
                line("root", "(([1] ([5-9]) | [2-9] [0-9]) | ([1-2] [0-9]{2} | [3] \"00\"))"),
                spaceLine()),
                flex("{\"type\": \"integer\", \"minimum\": 15, \"maximum\": 300}"));
    }

    // -------------------------------------------------- LC test "required props in original order" (:809-836)

    @Test
    public void requiredPropsKeepOriginalOrder() {
        assertEquals(lines(
                line("a-kv", "\"\\\"a\\\"\" space \":\" space string"),
                line("b-kv", "\"\\\"b\\\"\" space \":\" space string"),
                line("c-kv", "\"\\\"c\\\"\" space \":\" space string"),
                line("char", CHAR_RULE),
                line("root", "\"{\" space b-kv \",\" space c-kv \",\" space a-kv space \"}\""),
                spaceLine(),
                line("string", STRING_RULE)),
                flex("{\"type\": \"object\", "
                        + "\"properties\": {\"b\": {\"type\": \"string\"}, "
                        + "\"c\": {\"type\": \"string\"}, \"a\": {\"type\": \"string\"}}, "
                        + "\"required\": [\"a\", \"b\", \"c\"], "
                        + "\"additionalProperties\": false, \"definitions\": {}}"));
    }

    // -------------------------------------------------- LC test "top-level $ref" (:1138-1166)

    @Test
    public void topLevelRefResolvesLocalPointer() {
        assertEquals(lines(
                line("char", CHAR_RULE),
                line("ref-definitions-foo", "\"{\" space ref-definitions-foo-a-kv space \"}\""),
                line("root", "ref-definitions-foo"),
                spaceLine(),
                line("string", STRING_RULE)),
                flex("{\"$ref\": \"#/definitions/foo\", "
                        + "\"definitions\": {\"foo\": {\"type\": \"object\", "
                        + "\"properties\": {\"a\": {\"type\": \"string\"}}, "
                        + "\"required\": [\"a\"], \"additionalProperties\": false}}}"));
    }

    // -------------------------------------------------- LC test "anyOf $ref" (:1201-1237)

    @Test
    public void anyOfWithRefProperty() {
        assertEquals(lines(
                line("a", "string | number"),
                line("a-kv", "\"\\\"a\\\"\" space \":\" space a"),
                line("a-rest", "( \",\" space b-kv )?"),
                line("b", "b-0 | boolean"),
                line("b-0", "string"),
                line("b-kv", "\"\\\"b\\\"\" space \":\" space b"),
                line("boolean", "(\"true\" | \"false\")"),
                line("char", CHAR_RULE),
                line("decimal-part", DECIMAL_PART),
                line("integral-part", INTEGRAL_PART),
                line("number", NUMBER_RULE),
                line("root", "\"{\" space  (a-kv a-rest | b-kv )? space \"}\""),
                spaceLine(),
                line("string", STRING_RULE)),
                flex("{\"properties\": {"
                        + "\"a\": {\"anyOf\": [{\"type\": \"string\"}, {\"type\": \"number\"}]}, "
                        + "\"b\": {\"anyOf\": [{\"$ref\": \"#/properties/a/anyOf/0\"}, "
                        + "{\"type\": \"boolean\"}]}}, \"type\": \"object\"}"));
    }

    // -------------------------------------------------- LC test "non-string enum" (:483-492)

    @Test
    public void nonStringEnumConstants() {
        assertEquals(lines(
                line("root", "(\"\\\"red\\\"\" | \"\\\"amber\\\"\" | \"\\\"green\\\"\" "
                        + "| \"null\" | \"42\" | \"[\\\"foo\\\"]\")"),
                spaceLine()),
                flex("{\"enum\": [\"red\", \"amber\", \"green\", null, 42, [\"foo\"]]}"));
    }

    // -------------------------------------------------- LC test "exotic formats" (:337-361)

    @Test
    public void stringFormatsDateUuidTimeDateTime() {
        assertEquals(lines(
                line("date", "[0-9]{4} \"-\" ( \"0\" [1-9] | \"1\" [0-2] ) \"-\" "
                        + "( \"0\" [1-9] | [1-2] [0-9] | \"3\" [0-1] )"),
                line("date-string", "\"\\\"\" date \"\\\"\""),
                line("date-time", "date \"T\" time"),
                line("date-time-string", "\"\\\"\" date-time \"\\\"\""),
                line("root", "\"[\" space tuple-0 \",\" space uuid \",\" space tuple-2 "
                        + "\",\" space tuple-3 space \"]\""),
                spaceLine(),
                line("time", "([01] [0-9] | \"2\" [0-3]) \":\" [0-5] [0-9] \":\" [0-5] [0-9] "
                        + "( \".\" [0-9]{3} )? ( \"Z\" | ( \"+\" | \"-\" ) ( [01] [0-9] | \"2\" [0-3] ) "
                        + "\":\" [0-5] [0-9] )"),
                line("time-string", "\"\\\"\" time \"\\\"\""),
                line("tuple-0", "date-string"),
                line("tuple-2", "time-string"),
                line("tuple-3", "date-time-string"),
                line("uuid", "\"\\\"\" [0-9a-fA-F]{8} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} "
                        + "\"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{12} \"\\\"\"")),
                flex("{\"items\": ["
                        + "{\"format\": \"date\"}, {\"format\": \"uuid\"}, "
                        + "{\"format\": \"time\"}, {\"format\": \"date-time\"}]}"));
    }

    // -------------------------------------------------- LC test "additional props" (:909-926)

    @Test
    public void additionalPropertiesWithItemSchema() {
        assertEquals(lines(
                line("additional-kv", "string \":\" space additional-value"),
                line("additional-value", "\"[\" space (number (\",\" space number)*)? space \"]\""),
                line("char", CHAR_RULE),
                line("decimal-part", DECIMAL_PART),
                line("integral-part", INTEGRAL_PART),
                line("number", NUMBER_RULE),
                line("root", "\"{\" space  (additional-kv ( \",\" space additional-kv )* )? "
                        + "space \"}\""),
                spaceLine(),
                line("string", STRING_RULE)),
                flex("{\"type\": \"object\", \"additionalProperties\": "
                        + "{\"type\": \"array\", \"items\": {\"type\": \"number\"}}}"));
    }

    // -------------------------------------------------- row G8: the compact mode (derived from LC
    // "additional props": identical rules, the space rule becomes the empty alternate)

    @Test
    public void compactWhitespaceMode() {
        assertEquals(lines(
                line("additional-kv", "string \":\" space additional-value"),
                line("additional-value", "\"[\" space (number (\",\" space number)*)? space \"]\""),
                line("char", CHAR_RULE),
                line("decimal-part", DECIMAL_PART),
                line("integral-part", INTEGRAL_PART),
                line("number", NUMBER_RULE),
                line("root", "\"{\" space  (additional-kv ( \",\" space additional-kv )* )? "
                        + "space \"}\""),
                "space ::= ",
                line("string", STRING_RULE)),
                JsonSchemaToGbnf.gbnf("{\"type\": \"object\", \"additionalProperties\": "
                        + "{\"type\": \"array\", \"items\": {\"type\": \"number\"}}}",
                        JsonSchemaToGbnf.WhitespaceMode.COMPACT));
    }

    // -------------------------------------------------- LC test "mix of allOf, anyOf and $ref"
    // ("(similar to https://json.schemastore.org/tsconfig.json)", :1239-1281)

    @Test
    public void allOfAnyOfRefsMerge() {
        assertEquals(lines(
                line("a-kv", "\"\\\"a\\\"\" space \":\" space number"),
                line("b-kv", "\"\\\"b\\\"\" space \":\" space number"),
                line("c-kv", "\"\\\"c\\\"\" space \":\" space number"),
                line("d-kv", "\"\\\"d\\\"\" space \":\" space number"),
                line("d-rest", "( \",\" space c-kv )?"),
                line("decimal-part", DECIMAL_PART),
                line("integral-part", INTEGRAL_PART),
                line("number", NUMBER_RULE),
                line("root", "\"{\" space a-kv \",\" space b-kv "
                        + "( \",\" space ( d-kv d-rest | c-kv ) )? space \"}\""),
                spaceLine()),
                flex("{\"allOf\": [{\"$ref\": \"#/definitions/foo\"}, "
                        + "{\"$ref\": \"#/definitions/bar\"}, "
                        + "{\"anyOf\": [{\"$ref\": \"#/definitions/baz\"}, "
                        + "{\"$ref\": \"#/definitions/bam\"}]}], "
                        + "\"definitions\": {\"foo\": {\"properties\": {\"a\": {\"type\": \"number\"}}}, "
                        + "\"bar\": {\"properties\": {\"b\": {\"type\": \"number\"}}}, "
                        + "\"bam\": {\"properties\": {\"c\": {\"type\": \"number\"}}}, "
                        + "\"baz\": {\"properties\": {\"d\": {\"type\": \"number\"}}}}, "
                        + "\"type\": \"object\"}"));
    }

    // -------------------------------------------------- LC test "empty schema (any value)" (:317-334)
    // = the generic {@code json_object} grammar (row G3)

    @Test
    public void emptySchemaIsTheGenericJsonGrammar() {
        assertEquals(lines(
                line("array", "\"[\" space ( value (\",\" space value)* )? space \"]\""),
                line("boolean", "(\"true\" | \"false\")"),
                line("char", CHAR_RULE),
                line("decimal-part", DECIMAL_PART),
                line("integral-part", INTEGRAL_PART),
                line("null", "\"null\""),
                line("number", NUMBER_RULE),
                line("object", "\"{\" space ( string \":\" space value "
                        + "(\",\" space string \":\" space value)* )? space \"}\""),
                line("root", "value"),
                spaceLine(),
                line("string", STRING_RULE),
                line("value", "object | array | string | number | boolean | null")),
                JsonSchemaToGbnf.gbnf("{}", JsonSchemaToGbnf.WhitespaceMode.FLEXIBLE));
    }

    // -------------------------------------------------- the choice constraint (WP clause group;
    // no LC case: the alternation the spec clause states, "choice is an alternation of literals")

    @Test
    public void choiceConstraintIsAlternationOfEscapedLiterals() {
        List<String> choices = Arrays.asList("red", "amber", "green");
        Grammar grammar = GbnfBuilder.grammarOf(ConstraintSpec.choice(choices));
        // "root" plus the group the parser synthesizes for the "( ... )" alternation, plus "space"
        assertEquals(3, grammar.ruleCount());
        assertEquals(grammar.symbolId("root"), grammar.rootRuleIndex());
        List<GrammarElement> root = grammar.rule(grammar.rootRuleIndex());
        assertEquals(GrammarElement.Type.RULE_REF, root.get(0).type());
        assertEquals(GrammarElement.Type.END, root.get(1).type());
        int alternates = 0;
        for (GrammarElement element : grammar.rule(root.get(0).value())) {
            if (element.type() == GrammarElement.Type.ALT) {
                alternates++;
            }
        }
        assertEquals(2, alternates);
    }

    // -------------------------------------------------- ConstraintSpec: structural tags (SO-4)

    @Test
    public void structuralTagTextWithoutAStructuresArrayRefusesByName() {
        try {
            // a raw tag text is not the legacy payload: the payload needs its structures array
            GbnfBuilder.grammarOf(ConstraintSpec.structuralTag("<bos>"));
            fail("a payload without structures refuses");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("not JSON"));
        }
        try {
            GbnfBuilder.grammarOf(ConstraintSpec.structuralTag(
                    "{\"structures\": [{\"begin\": \"<bos>\", \"schema\": {}, \"end\": \"<eos>\"}],"
                            + " \"triggers\": [\"<bos>\"]}"));
            fail("trigger-gated free text is pending");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("triggers"));
        }
    }

    // -------------------------------------------------- refusal: remote $ref (WP spec clause;
    // llama.cpp's _visit_ref helper refuses non-local pointers, no golden case in the LC tests)

    @Test
    public void remoteRefIsRefused() {
        try {
            flex("{\"$ref\": \"https://example.com/common.json#/some/rules\"}");
            fail("remote $ref must be refused");
        } catch (IllegalArgumentException e) {
            String message = e.getMessage();
            assertTrue(message, message.contains("JSON schema conversion failed:"));
            assertTrue(message, message.contains("unsupported $ref "
                    + "https://example.com/common.json#/some/rules"));
            assertTrue(message, message.contains("only references into the same document"));
        }
    }

    // -------------------------------------------------- LC test "unknown type" (:299-306, FAILURE)

    @Test
    public void unknownTypeIsRefused() {
        try {
            flex("{\"type\": \"kaboom\"}");
            fail("an unknown type must be refused");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("unrecognized type kaboom"));
        }
    }

    // -------------------------------------------------- LC test "unanchored regexp" (:1448-1461)
    // an unanchored pattern degrades the property to the generic string rule

    @Test
    public void unanchoredPatternDegradesToString() {
        assertEquals(lines(
                line("char", CHAR_RULE),
                line("root", "string"),
                spaceLine(),
                line("string", STRING_RULE)),
                flex("{\"type\": \"string\", \"pattern\": \"[0-9]+\"}"));
    }

    // -------------------------------------------------- LC test "regexp with unsupported shorthand"
    // (:1464-1477): llama.cpp's comment — "the rules of the partial conversion (here "root-0")
    // must not leak into the grammar"

    @Test
    public void unsupportedShorthandDegradesWithoutLeakingPartialRules() {
        String grammar = flex("{\"type\": \"string\", \"pattern\": \"^[0-9]{3}\\\\w$\"}");
        assertEquals(lines(
                line("char", CHAR_RULE),
                line("root", "string"),
                spaceLine(),
                line("string", STRING_RULE)),
                grammar);
        assertFalse(grammar, grammar.contains("root-0"));
        assertFalse(grammar, grammar.contains("root-1"));
    }

    // -------------------------------------------------- LC test "unsupported regexp in a property"
    // (:1517-1536): only the property with the bad pattern degrades ("^(?=a)a$", a lookahead)
    // and it degrades to a rule of the property's own name

    @Test
    public void unsupportedLookaheadDegradesOnlyThatProperty() {
        assertEquals(lines(
                line("a", "string"),
                line("a-kv", "\"\\\"a\\\"\" space \":\" space a"),
                line("char", CHAR_RULE),
                line("root", "\"{\" space a-kv space \"}\""),
                spaceLine(),
                line("string", STRING_RULE)),
                flex("{\"type\": \"object\", "
                        + "\"properties\": {\"a\": {\"type\": \"string\", "
                        + "\"pattern\": \"^(?=a)a$\"}}, "
                        + "\"required\": [\"a\"], \"additionalProperties\": false}"));
    }

    // -------------------------------------------------- LC test "regexp with unbalanced parentheses"
    // (:1506-1514, FAILURE): "a regexp that is invalid under any flavor is still an error"

    @Test
    public void invalidPatternFailsTheConversion() {
        try {
            flex("{\"type\": \"string\", \"pattern\": \"^(a$\"}");
            fail("an invalid pattern must be refused");
        } catch (IllegalArgumentException e) {
            String message = e.getMessage();
            assertTrue(message, message.contains("JSON schema conversion failed:"));
            assertTrue(message, message.contains("Invalid pattern ^(a$:"));
            assertTrue(message, message.contains("unbalanced parentheses"));
        }
    }

    // -------------------------------------------------- Lark refusals (WP spec clause: Lark is refused
    // where it uses %import): a %-directive is no GBNF rule line, so the parse refuses it

    @Test
    public void larkPercentImportIsRefused() {
        try {
            GbnfParser.parse("%import common.Lexer\nroot ::= \"x\"");
            fail("a Lark %-import must be refused");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("expecting"));
        }
    }
}

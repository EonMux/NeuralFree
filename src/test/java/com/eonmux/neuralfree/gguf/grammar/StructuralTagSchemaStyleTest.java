package com.eonmux.neuralfree.gguf.grammar;

import org.junit.Test;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The structural-tag {@code json_schema} format's {@code style}, {@code any_order} and {@code max_whitespace_cnt}
 * ({@link JsonSchemaOptions}, {@link XmlParameterRules}), and the deprecated {@code qwen_xml_parameter} format. Every
 * accept/reject verdict is xgrammar 0.2.8's ({@code GrammarMatcher.accept_string} then {@code is_terminated}) for
 * the same payload and text, but the one difference {@link #formatStringsAreTheirRawText} names.
 */
public class StructuralTagSchemaStyleTest {

    private static final String DSML = "｜DSML｜";

    /** {@code name} (string) and {@code age} (integer) required, {@code tag} an optional string enum. */
    private static final String PERSON = "{\"type\": \"object\", \"properties\": {\"name\": {\"type\": \"string\"},"
            + " \"age\": {\"type\": \"integer\"}, \"tag\": {\"type\": \"string\", \"enum\": [\"a\", \"b\"]}},"
            + " \"required\": [\"name\", \"age\"]}";

    /** One property per value kind, {@code p} (a pattern) required. */
    private static final String VALUES = "{\"type\": \"object\", \"properties\": {"
            + "\"p\": {\"type\": \"string\", \"pattern\": \"^a\\\"b+$\"},"
            + " \"m\": {\"type\": \"string\", \"maxLength\": 2},"
            + " \"f\": {\"type\": \"string\", \"format\": \"date\"},"
            + " \"n\": {\"type\": [\"string\", \"null\"]},"
            + " \"c\": {\"const\": \"q\"},"
            + " \"arr\": {\"type\": \"array\", \"items\": {\"type\": \"integer\"}},"
            + " \"o\": {\"type\": \"object\", \"properties\": {\"x\": {\"type\": \"string\"}}, \"required\": [\"x\"]},"
            + " \"u\": {\"anyOf\": [{\"type\": \"string\"}, {\"type\": \"integer\"}]},"
            + " \"e\": {\"enum\": [1, \"z\", null]},"
            + " \"x\": {}}, \"required\": [\"p\"]}";

    /** {@code a}, {@code b} and {@code c} integers, {@code b} required. */
    private static final String ORDERED = "{\"type\": \"object\", \"properties\": {\"a\": {\"type\": \"integer\"},"
            + " \"b\": {\"type\": \"integer\"}, \"c\": {\"type\": \"integer\"}}, \"required\": [\"b\"]";

    /** {@code a} required, {@code b} an integer array and {@code c} a string optional. */
    private static final String MEMBERS = "{\"type\": \"object\", \"properties\": {\"a\": {\"type\": \"integer\"},"
            + " \"b\": {\"type\": \"array\", \"items\": {\"type\": \"integer\"}}, \"c\": {\"type\": \"string\"}},"
            + " \"required\": [\"a\"]";

    private static String schema(String schema, String extra) {
        return "{\"type\": \"json_schema\", \"json_schema\": " + schema + (extra.isEmpty() ? "" : ", " + extra) + "}";
    }

    private static String style(String schema, String style) {
        return schema(schema, "\"style\": \"" + style + "\"");
    }

    private static Grammar grammar(String format) {
        return GbnfBuilder.grammarOf(ConstraintSpec.structuralTag("{\"format\": " + format + "}"));
    }

    private static void accepts(String format, String... texts) {
        Grammar grammar = grammar(format);
        for (String text : texts) {
            assertTrue("accepts " + text, GrammarMatcher.matchesUtf8(grammar, text));
        }
    }

    private static void rejects(String format, String... texts) {
        Grammar grammar = grammar(format);
        for (String text : texts) {
            assertFalse("rejects " + text, GrammarMatcher.matchesUtf8(grammar, text));
        }
    }

    private static void assertRefused(String format, String messagePart) {
        try {
            grammar(format);
            throw new AssertionError("expected a refusal for " + format);
        } catch (IllegalArgumentException e) {
            assertTrue("the refusal names the gap: " + e.getMessage(), e.getMessage().contains(messagePart));
        }
    }

    private static String qwen(String name, String value) {
        return "<parameter=" + name + ">" + value + "</parameter>";
    }

    private static String dsml(String name, boolean string, String value) {
        return "<" + DSML + "parameter name=\"" + name + "\" string=\"" + string + "\">" + value + "</" + DSML
                + "parameter>";
    }

    // ------------------------------------------------------------------ the styles

    @Test
    public void qwenXmlWritesTheTopLevelPropertiesAsParametersInSchemaOrder() {
        String format = style(PERSON, "qwen_xml");
        accepts(format, qwen("name", "Bob") + qwen("age", "3"),
                "\n" + qwen("name", "\nBob <x>\n") + "\n" + qwen("age", "\n3\n") + "\n",
                qwen("name", "Bob") + qwen("age", "3") + qwen("tag", "a"),
                qwen("name", "Bob") + qwen("age", "3") + qwen("tag", " b\n"),
                qwen("name", "") + qwen("age", "-5"));
        rejects(format, qwen("name", "Bob") + qwen("age", "3") + qwen("tag", "c"),
                qwen("age", "3") + qwen("name", "Bob"),
                qwen("name", "Bob"),
                qwen("name", "Bo") + "b</parameter>" + qwen("age", "3"),
                "");
    }

    @Test
    public void theDeprecatedQwenXmlParameterFormatIsTheQwenXmlStyle() {
        String format = "{\"type\": \"qwen_xml_parameter\", \"json_schema\": " + PERSON + "}";
        accepts(format, qwen("name", "Bob") + qwen("age", "3"));
        rejects(format, qwen("age", "3") + qwen("name", "Bob"), "{\"name\": \"Bob\", \"age\": 3}");
    }

    @Test
    public void minimaxXmlNamesTheParameterInAnAttribute() {
        String format = style(PERSON, "minimax_xml");
        accepts(format, "<parameter name=\"name\">Bob</parameter><parameter name=\"age\">3</parameter>",
                "<parameter name=\"name\">Bob</parameter><parameter name=\"age\">3</parameter>"
                        + "<parameter name=\"tag\"> b\n</parameter>");
        rejects(format, "<parameter name=\"age\">3</parameter><parameter name=\"name\">Bob</parameter>",
                qwen("name", "Bob") + qwen("age", "3"));
    }

    @Test
    public void glmXmlWritesKeyAndValueTags() {
        String format = style(PERSON, "glm_xml");
        accepts(format, "<arg_key>name</arg_key><arg_value>Bob</arg_value><arg_key>age</arg_key>\n<arg_value>3"
                        + "</arg_value>",
                "<arg_key>name</arg_key> <arg_value>Bob</arg_value>\n<arg_key>age</arg_key><arg_value> 3 </arg_value>"
                        + "<arg_key>tag</arg_key><arg_value>b</arg_value>");
        rejects(format, "<arg_key>age</arg_key><arg_value>3</arg_value>");
    }

    @Test
    public void deepseekXmlDeclaresWhetherTheValueIsAString() {
        String format = style(PERSON, "deepseek_xml");
        accepts(format, dsml("name", true, "Bob") + dsml("age", false, "3"),
                dsml("name", true, "Bob") + dsml("age", false, " 3\n") + dsml("tag", true, "a"));
        rejects(format, dsml("name", false, "Bob") + dsml("age", false, "3"),
                dsml("name", true, "Bob") + dsml("age", true, "3"),
                dsml("name", true, "Bob") + dsml("age", false, "3") + dsml("tag", true, " a"));
    }

    @Test
    public void deepseekV41XmlSpacesItsTags() {
        String format = style(PERSON, "deepseek_v4_1_xml");
        String spaced = DSML + " ";
        accepts(format, "<" + spaced + "parameter name=\"name\" string=\"true\">Bob</" + spaced + "parameter><"
                + spaced + "parameter name=\"age\" string=\"false\">3</" + spaced + "parameter>");
        rejects(format, dsml("name", true, "Bob") + dsml("age", false, "3"));
    }

    // ------------------------------------------------------------------ values

    @Test
    public void eachValueKindIsWrittenRawOrAsJson() {
        String format = style(VALUES, "qwen_xml");
        String p = qwen("p", "a\"b");
        accepts(format, qwen("p", "a\"bb"), p + qwen("m", "xy"), p + qwen("f", "2024-01-31"),
                p + qwen("n", "null"), p + qwen("n", "hello"), p + qwen("c", "q"), p + qwen("arr", "[1,2]"),
                p + qwen("o", "{\"x\":\"y\"}"), p + qwen("u", "12"), p + qwen("u", "text"), p + qwen("e", "1"),
                p + qwen("e", "z"), p + qwen("e", "null"), p + qwen("x", "anything"), p + qwen("x", "[1]"),
                p + qwen("x", "{\"k\":true}"));
        rejects(format, qwen("p", "ab"), p + qwen("m", "xyz"), p + qwen("f", "2024-13-31"),
                p + qwen("c", "\"q\""), p + qwen("arr", "[1,\"a\"]"), p + qwen("o", "{\"x\":1}"),
                p + qwen("e", "\"z\""));
    }

    /**
     * A {@code format} string is its raw text. A quoted e-mail local part is RFC 5321's {@code "a b"@x.io}, where
     * xgrammar keeps the JSON string's escapes ({@code \"a b\"@x.io}).
     */
    @Test
    public void formatStringsAreTheirRawText() {
        String format = style("{\"type\": \"object\", \"properties\": {\"u\": {\"type\": \"string\", \"format\":"
                + " \"uuid\"}, \"m\": {\"type\": \"string\", \"format\": \"email\"}, \"j\": {\"type\": \"string\","
                + " \"format\": \"json-pointer\"}}}", "qwen_xml");
        accepts(format, qwen("u", "123e4567-e89b-12d3-a456-426614174000"), qwen("m", "a.b@x.io"),
                qwen("m", "\"a b\"@x.io"), qwen("j", "/a\"b/c~1"), qwen("j", "/a\\b"));
        rejects(format, qwen("u", "\"123e4567-e89b-12d3-a456-426614174000\""), qwen("m", "\\\"a b\\\"@x.io"),
                qwen("j", "/a~2"));
    }

    @Test
    public void deepseekXmlSplitsAUnionByItsMembersKind() {
        String format = style(VALUES, "deepseek_xml");
        String p = dsml("p", true, "a\"b");
        accepts(format, p + dsml("n", false, "null"), p + dsml("n", true, "null"), p + dsml("u", false, "5"),
                p + dsml("u", true, "5"), p + dsml("e", true, "z"), p + dsml("e", false, "1"),
                p + dsml("x", false, "true"), p + dsml("x", true, "true"));
        rejects(format, p + dsml("e", false, "z"), dsml("p", false, "a\"b"));
    }

    @Test
    public void additionalPropertiesTakeIdentifierNames() {
        String format = style("{\"type\": \"object\", \"properties\": {\"a\": {\"type\": \"integer\"}},"
                + " \"additionalProperties\": {\"type\": \"boolean\"}}", "qwen_xml");
        accepts(format, "", qwen("a", "1"), qwen("a", "1") + qwen("zz", "true") + qwen("_q9", "false"),
                qwen("zz", "true"));
        rejects(format, qwen("a", "1") + qwen("9z", "true"), qwen("a", "1") + qwen("zz", "1"));
    }

    @Test
    public void aSchemaOfAnyObjectTakesAnyParameters() {
        String format = "{\"type\": \"json_schema\", \"json_schema\": true, \"style\": \"qwen_xml\"}";
        accepts(format, "", qwen("k", "v"), qwen("k", "[1, 2]") + " " + qwen("j", "x"));
        rejects(format, qwen("k-1", "v"));
    }

    @Test
    public void aNonObjectTopLevelIsOneValue() {
        accepts(style("{\"type\": \"string\"}", "qwen_xml"), "hello <b>", "");
        rejects(style("{\"type\": \"string\"}", "qwen_xml"), "a</parameter>");
        accepts(style("{\"type\": \"integer\"}", "qwen_xml"), "12");
        rejects(style("{\"type\": \"integer\"}", "qwen_xml"), " 12", "x");
    }

    @Test
    public void topLevelUnionsAndReferencesReachTheirObjects() {
        String union = style("{\"anyOf\": [{\"type\": \"object\", \"properties\": {\"a\": {\"type\": \"integer\"}},"
                + " \"required\": [\"a\"]}, {\"type\": \"object\", \"properties\": {\"b\": {\"type\": \"string\"}},"
                + " \"required\": [\"b\"]}]}", "qwen_xml");
        accepts(union, qwen("a", "1"), qwen("b", "x"));
        rejects(union, qwen("a", "x"), "");
        String reference = style("{\"$ref\": \"#/$defs/o\", \"$defs\": {\"o\": {\"type\": \"object\", \"properties\":"
                + " {\"a\": {\"type\": \"string\"}}, \"required\": [\"a\"]}}}", "qwen_xml");
        accepts(reference, qwen("a", "x"));
        rejects(reference, "");
    }

    @Test
    public void requiredParametersKeepTheirPlaceAmongOptionalOnes() {
        String format = style(ORDERED + "}", "qwen_xml");
        accepts(format, qwen("a", "1") + qwen("b", "2"), qwen("b", "2"), qwen("b", "2") + qwen("c", "3"),
                qwen("a", "1") + qwen("b", "2") + qwen("c", "3"));
        rejects(format, qwen("a", "1") + qwen("c", "3"), qwen("b", "2") + qwen("a", "1"));
    }

    @Test
    public void propertyCountsBoundTheParameters() {
        String format = style(ORDERED + ", \"minProperties\": 2, \"maxProperties\": 2}", "qwen_xml");
        accepts(format, qwen("a", "1") + qwen("b", "2"), qwen("b", "2") + qwen("c", "3"));
        rejects(format, qwen("b", "2"), qwen("a", "1") + qwen("b", "2") + qwen("c", "3"));
    }

    // ------------------------------------------------------------------ any_order

    @Test
    public void anyOrderTakesParametersInAnyOrderAndRepeated() {
        String format = schema(ORDERED + "}", "\"style\": \"qwen_xml\", \"any_order\": true");
        accepts(format, qwen("b", "2") + qwen("a", "1"), qwen("c", "2") + qwen("c", "1"), qwen("b", "2"),
                "\n" + qwen("a", "1") + "\n" + qwen("b", "2") + "\n" + qwen("c", "3") + "\n" + qwen("a", "4") + "\n");
        rejects(format, "");
        String bounded = schema(ORDERED + ", \"maxProperties\": 2}", "\"style\": \"qwen_xml\", \"any_order\": true");
        accepts(bounded, qwen("b", "2") + qwen("a", "1"));
        rejects(bounded, qwen("a", "1") + qwen("b", "2") + qwen("c", "3"));
    }

    @Test
    public void anyOrderJsonObjectsBoundOnlyTheMemberCount() {
        String format = schema(MEMBERS + "}", "\"any_order\": true");
        accepts(format, "{\"a\":1}", "{\"c\":\"x\",\"a\":1}", "{\"b\":[1],\"c\":\"x\"}", "{\"a\":1,\"a\":2}",
                "{\"a\":1, \"b\":[2]}");
        rejects(format, "{}", "{\"d\":1}");
        accepts(schema("{\"type\": \"object\", \"properties\": {\"a\": {\"type\": \"integer\"}}}",
                "\"any_order\": true"), "{}", "{\"a\":1}", "{\"a\":1,\"a\":2}");
        String counted = schema(MEMBERS + ", \"minProperties\": 2, \"maxProperties\": 3}", "\"any_order\": true");
        accepts(counted, "{\"c\":\"x\",\"a\":1}", "{\"a\":1,\"a\":2,\"a\":3}");
        rejects(counted, "{\"a\":1}", "{\"a\":1,\"a\":2,\"a\":3,\"a\":4}");
    }

    @Test
    public void anyOrderReachesNestedObjects() {
        String format = schema("{\"type\": \"object\", \"properties\": {\"o\": {\"type\": \"object\", \"properties\":"
                + " {\"x\": {\"type\": \"integer\"}, \"y\": {\"type\": \"integer\"}}, \"required\": [\"x\", \"y\"]}},"
                + " \"required\": [\"o\"]}", "\"any_order\": true");
        accepts(format, "{\"o\":{\"y\":1,\"x\":2}}");
        rejects(format, "{\"o\":{\"x\":1}}");
    }

    // ------------------------------------------------------------------ max_whitespace_cnt

    @Test
    public void maxWhitespaceCountLimitsEveryJsonWhitespaceRun() {
        String format = schema(MEMBERS + "}", "\"max_whitespace_cnt\": 2");
        accepts(format, "{\"a\":1}", "{  \"a\":1}", "{\"a\" :  1}", "{\"a\":1,  \"b\":[1,  2]  }", "{\n\n\"a\":1}");
        rejects(format, "{   \"a\":1}");
    }

    @Test
    public void maxWhitespaceCountLimitsEveryXmlWhitespaceRun() {
        String format = schema(PERSON, "\"style\": \"qwen_xml\", \"max_whitespace_cnt\": 2");
        accepts(format, qwen("name", "Bob") + "  " + qwen("age", "  3  ") + "  ",
                qwen("name", "   Bob   ") + qwen("age", "3"));
        rejects(format, qwen("name", "Bob") + "   " + qwen("age", "3"), qwen("name", "Bob") + qwen("age", "   3"));
    }

    // ------------------------------------------------------------------ property order

    @Test
    public void aStructuralTagSchemaKeepsItsPropertyOrder() {
        String format = schema("{\"type\": \"object\", \"properties\": {\"b\": {\"type\": \"integer\"},"
                + " \"a\": {\"type\": \"integer\"}}, \"required\": [\"b\", \"a\"]}", "");
        accepts(format, "{\"b\":1,\"a\":2}");
        rejects(format, "{\"a\":2,\"b\":1}");
    }

    // ------------------------------------------------------------------ refusals

    @Test
    public void unsupportedStylesAndKnobsAreRefusedByName() {
        assertRefused(style(PERSON, "cohere_xml"), "style 'cohere_xml' is not supported");
        assertRefused(style(PERSON, "kimi_k3_xml"), "style 'kimi_k3_xml' is not supported");
        assertRefused(style(PERSON, "yaml"), "style 'yaml' is not recognized");
        assertRefused(schema(PERSON, "\"max_whitespace_cnt\": 0"), "max_whitespace_cnt must be an integer from 1");
        assertRefused(schema(PERSON, "\"max_whitespace_cnt\": 1.5"), "max_whitespace_cnt must be an integer");
        assertRefused(schema(PERSON, "\"any_order\": \"yes\""), "any_order must be a boolean");
        assertRefused(style("{\"type\": \"object\", \"patternProperties\": {\"^q_\": {\"type\": \"integer\"}}}",
                "qwen_xml"), "patternProperties of root are not supported in the qwen_xml style");
        assertRefused(style("{\"type\": \"object\", \"propertyNames\": {\"pattern\": \"^a\"}}", "glm_xml"),
                "propertyNames of root are not supported in the glm_xml style");
        assertRefused(style("{\"type\": \"object\", \"properties\": {\"a\": {\"type\": \"string\", \"pattern\":"
                + " \"(?=a)a\"}}}", "qwen_xml"), "pattern (?=a)a is not supported");
    }
}

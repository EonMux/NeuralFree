package com.eonmux.neuralfree.gguf.grammar;

import com.eonmux.neuralfree.utils.json.Json;
import org.junit.Test;

import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The structured-output request options over the grammar compiler: {@code disable_any_whitespace} (the compact
 * mode), an explicit {@code whitespace_pattern} regex as the {@code space} rule, and
 * {@code disable_additional_properties} closing every schema object that declares properties (vLLM's
 * {@code StructuredOutputsParams} and guidance's {@code process_for_additional_properties}). Every grammar is
 * driven over JSON text with {@link GrammarMatcher}, so a case fails when the option stops admitting a value it
 * should or admits one it should not.
 */
public class StructuredOutputOptionsTest {

    private static final String OBJECT_SCHEMA =
            "{\"type\": \"object\", \"properties\": {\"a\": {\"type\": \"boolean\"}}, \"required\": [\"a\"]}";

    private static Grammar grammar(ConstraintSpec spec) {
        return GbnfBuilder.grammarOf(spec);
    }

    private static void accepts(Grammar grammar, String... texts) {
        for (String text : texts) {
            assertTrue("should accept " + text, GrammarMatcher.matches(grammar, text));
        }
    }

    private static void rejects(Grammar grammar, String... texts) {
        for (String text : texts) {
            assertFalse("should reject " + text, GrammarMatcher.matches(grammar, text));
        }
    }

    // -------------------------------------------------- disable_any_whitespace

    @Test
    public void disableAnyWhitespaceWritesTheCompactForm() {
        ConstraintSpec spec = ConstraintSpec.jsonSchema(OBJECT_SCHEMA, null).withJsonOptions(true, null, false);
        assertEquals(JsonSchemaToGbnf.WhitespaceMode.COMPACT, spec.whitespace());
        Grammar compact = grammar(spec);
        accepts(compact, "{\"a\":true}", "{\"a\":false}");
        rejects(compact, "{\"a\": true}", "{\n\"a\": true}", "{ \"a\":false }");
    }

    @Test
    public void disableAnyWhitespaceAppliesToJsonObject() {
        Grammar compact = grammar(ConstraintSpec.jsonObject(null).withJsonOptions(true, null, false));
        accepts(compact, "{\"a\":[1,{}]}");
        rejects(compact, "{ }", "{\"a\": 1}");
    }

    // -------------------------------------------------- whitespace_pattern

    @Test
    public void whitespacePatternBecomesTheSpaceRule() {
        ConstraintSpec spec = ConstraintSpec.jsonSchema(OBJECT_SCHEMA, null)
                .withJsonOptions(false, "[ ]*", false);
        assertEquals("[ ]*", spec.whitespacePattern());
        Grammar spaced = grammar(spec);
        accepts(spaced, "{\"a\":true}", "{\"a\": true}", "{\"a\" :true}", "{\"a\" : true}");
        rejects(spaced, "{\n\"a\":true}", "{\t\"a\": true}", "{\r\"a\": true}");
        String gbnf = JsonSchemaToGbnf.gbnf(OBJECT_SCHEMA,
                JsonSchemaOptions.json(null).withWhitespacePattern("[ ]*"));
        assertTrue(gbnf, gbnf.contains("space ::= space-pattern"));
        assertTrue(gbnf, gbnf.contains("space-pattern ::= ([ ]*)"));
    }

    @Test
    public void whitespacePatternAdmitsTheWhitespaceItNames() {
        Grammar newlines = grammar(ConstraintSpec.jsonSchema(OBJECT_SCHEMA, null)
                .withJsonOptions(false, "[\\n\\t ]*", false));
        accepts(newlines, "{\n\t\"a\":\ttrue }", "{\"a\":\ntrue}");
        rejects(newlines, "{\r\"a\":true}");
    }

    @Test
    public void whitespacePatternWinsOverTheCompactMode() {
        Grammar newlines = grammar(ConstraintSpec.jsonSchema(OBJECT_SCHEMA, null)
                .withJsonOptions(true, "\\n*", false));
        accepts(newlines, "{\n\"a\":\ntrue}");
        rejects(newlines, "{\"a\": true}");
    }

    @Test
    public void whitespacePatternRefusesInvalidAndUnsupportedRegexesByName() {
        String lookahead = refusal(ConstraintSpec.jsonSchema(OBJECT_SCHEMA, null)
                .withJsonOptions(false, "(?=a)", false));
        assertTrue(lookahead, lookahead.contains("whitespace_pattern (?=a) uses an unsupported construct:"));
        assertTrue(lookahead, lookahead.contains("lookahead assertion"));
        String invalid = refusal(ConstraintSpec.jsonSchema(OBJECT_SCHEMA, null)
                .withJsonOptions(false, "[", false));
        assertTrue(invalid, invalid.contains("whitespace_pattern [ is not a valid regex:"));
        try {
            JsonSchemaOptions.json(null).withWhitespacePattern("");
            fail("an empty whitespace_pattern must be refused");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("whitespace_pattern must not be empty"));
        }
    }

    private static String refusal(ConstraintSpec spec) {
        try {
            GbnfBuilder.grammarOf(spec);
            fail("the constraint should be refused");
        } catch (IllegalArgumentException e) {
            return e.getMessage();
        }
        return null;
    }

    // -------------------------------------------------- disable_additional_properties

    @Test
    public void disableAdditionalPropertiesClosesSchemaPositionsOnly() {
        String schema = "{\"type\": \"object\", \"properties\": {"
                + "\"nested\": {\"type\": \"object\", \"properties\": {\"x\": {\"type\": \"string\"}}},"
                + "\"literal\": {\"const\": {\"type\": \"Feature\", \"properties\": {\"name\": \"HQ\"}}},"
                + "\"choices\": {\"enum\": [{\"type\": \"Feature\", \"properties\": {\"name\": \"HQ\"}}]}},"
                + "\"$defs\": {\"item\": {\"type\": \"object\", \"properties\": {\"y\": {\"type\": \"boolean\"}}}}}";
        Map<?, ?> document = (Map<?, ?>) Json.parse(schema, Json.Limits.UNTRUSTED);
        Map<?, ?> closed = (Map<?, ?>) JsonSchemaAdditionalProperties.closed(document);

        assertEquals(Boolean.FALSE, closed.get("additionalProperties"));
        assertFalse("the walk copies its input", document.containsKey("additionalProperties"));
        Map<?, ?> properties = (Map<?, ?>) closed.get("properties");
        assertEquals(Boolean.FALSE, ((Map<?, ?>) properties.get("nested")).get("additionalProperties"));
        Map<?, ?> defs = (Map<?, ?>) closed.get("$defs");
        assertEquals(Boolean.FALSE, ((Map<?, ?>) defs.get("item")).get("additionalProperties"));
        Map<?, ?> literal = (Map<?, ?>) ((Map<?, ?>) properties.get("literal")).get("const");
        assertNull("a const value is not a schema position", literal.get("additionalProperties"));
        List<?> choices = (List<?>) ((Map<?, ?>) properties.get("choices")).get("enum");
        assertNull("an enum value is not a schema position",
                ((Map<?, ?>) choices.get(0)).get("additionalProperties"));
    }

    @Test
    public void disableAdditionalPropertiesKeepsAnExplicitValue() {
        Map<?, ?> open = (Map<?, ?>) JsonSchemaAdditionalProperties.closed(Json.parse(
                "{\"type\": \"object\", \"properties\": {\"a\": {}}, \"additionalProperties\": true}",
                Json.Limits.UNTRUSTED));
        assertEquals(Boolean.TRUE, open.get("additionalProperties"));
        Map<?, ?> closed = (Map<?, ?>) JsonSchemaAdditionalProperties.closed(Json.parse(
                "{\"type\": \"object\", \"properties\": {\"a\": {}}, \"additionalProperties\": false}",
                Json.Limits.UNTRUSTED));
        assertEquals(Boolean.FALSE, closed.get("additionalProperties"));
    }

    @Test
    public void disableAdditionalPropertiesMatchesTheExplicitlyClosedSchema() {
        String closed = JsonSchemaToGbnf.gbnf(OBJECT_SCHEMA,
                JsonSchemaOptions.json(null).withAdditionalPropertiesDisabled(true));
        String explicit = JsonSchemaToGbnf.gbnf(OBJECT_SCHEMA.substring(0, OBJECT_SCHEMA.length() - 1)
                + ", \"additionalProperties\": false}", JsonSchemaOptions.json(null));
        assertEquals(explicit, closed);
        ConstraintSpec spec = ConstraintSpec.jsonSchema(OBJECT_SCHEMA, null).withJsonOptions(false, null, true);
        assertTrue(spec.disableAdditionalProperties());
        Grammar grammar = grammar(spec);
        accepts(grammar, "{\"a\":true}", "{\"a\": false}");
        rejects(grammar, "{\"a\":true,\"b\":false}");
    }

    // -------------------------------------------------- the request-shape guard

    @Test
    public void jsonOptionsAreRefusedOnOtherConstraintKinds() {
        try {
            ConstraintSpec.grammar("root ::= \"x\"").withJsonOptions(true, null, false);
            fail("the options must not attach to a grammar");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("apply to json and json_object constraints only"));
            assertTrue(e.getMessage(), e.getMessage().contains("disable_any_whitespace"));
        }
        ConstraintSpec untouched = ConstraintSpec.grammar("root ::= \"x\"").withJsonOptions(false, null, false);
        assertEquals(ConstraintSpec.Kind.GBNF, untouched.kind());
        assertEquals("root ::= \"x\"", untouched.text());
        try {
            ConstraintSpec.jsonObject(null).withJsonOptions(false, "", false);
            fail("an empty whitespace_pattern must be refused");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("whitespace_pattern must not be empty"));
        }
    }
}

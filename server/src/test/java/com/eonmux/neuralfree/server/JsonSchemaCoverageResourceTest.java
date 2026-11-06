package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.math.BigDecimal;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The xgrammar-level JSON-schema keywords over the real service: {@code response_format}, {@code guided_json} and
 * {@code structured_outputs.json} schemas with a string format, number bounds and key/count constraints drive the
 * synthetic llama's sampling, so the completion text is a value of the schema; the keyword combinations no grammar
 * can honour, and references out of the document, are 400s naming the keyword.
 */
class JsonSchemaCoverageResourceTest {

    private static final Pattern IPV4 =
            Pattern.compile("(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])(\\.(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])){3}");

    private static EngineService service;
    private static OpenAiResource resource;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig config = ServerTestSupport.pagedConfig(256, 2);
        service = new EngineService(model.toString(), config);
        resource = new OpenAiResource(service);
    }

    @AfterAll
    static void close() {
        service.close();
    }

    /** The parsed completion text of a request carrying {@code constraint}, which must end on its own. */
    private static Object constrained(String constraint) {
        return Json.parse(constrainedText(constraint), Json.Limits.UNTRUSTED);
    }

    /** The completion text of a request carrying {@code constraint}, which must end on its own. */
    private static String constrainedText(String constraint) {
        Response response = resource.completions("{\"prompt\": \"value:\", \"max_tokens\": 200, \"temperature\": 0, "
                + constraint + "}");
        assertEquals(200, response.getStatus());
        JsonView choice = JsonView.of(((List<?>) view(response.getEntity()).raw("choices")).get(0), "choices[0]");
        assertEquals("stop", choice.str("finish_reason"), "the grammar completes the value: " + choice.str("text"));
        return choice.str("text");
    }

    @Test
    void aFormatSchemaYieldsAValueOfTheFormat() {
        Object value = constrained("\"response_format\": {\"type\": \"json_schema\", \"json_schema\": {\"name\": \"ip\","
                + " \"schema\": {\"type\": \"string\", \"format\": \"ipv4\"}}}");
        assertTrue(value instanceof String && IPV4.matcher((String) value).matches(), "an IPv4 address: " + value);
    }

    @Test
    void numberBoundsHoldOnTheSampledNumber() {
        Object value = constrained("\"guided_json\": {\"type\": \"number\", \"exclusiveMinimum\": 0.25,"
                + " \"maximum\": 0.5}");
        BigDecimal number = new BigDecimal(String.valueOf(value));
        assertTrue(number.compareTo(new BigDecimal("0.25")) > 0 && number.compareTo(new BigDecimal("0.5")) <= 0,
                "a number in (0.25, 0.5]: " + value);
    }

    @Test
    void keyAndCountConstraintsHoldOnTheSampledObject() {
        Object value = constrained("\"structured_outputs\": {\"json\": {\"type\": \"object\","
                + " \"propertyNames\": {\"enum\": [\"a\", \"b\"]}, \"additionalProperties\": {\"type\": \"boolean\"},"
                + " \"minProperties\": 1, \"maxProperties\": 2}}");
        assertTrue(value instanceof Map, "an object: " + value);
        Map<?, ?> object = (Map<?, ?>) value;
        assertTrue(!object.isEmpty() && object.size() <= 2, "one or two members: " + value);
        for (Map.Entry<?, ?> entry : object.entrySet()) {
            assertTrue(entry.getKey().equals("a") || entry.getKey().equals("b"), "a named key: " + value);
            assertTrue(entry.getValue() instanceof Boolean, "a boolean member: " + value);
        }
    }

    @Test
    void ungrammaticalKeywordCombinationsAreRefusedByName() {
        String combined = errorOf(() -> resource.completions("{\"prompt\": \"x\", \"guided_json\": {\"type\": \"object\","
                + " \"patternProperties\": {\"^a\": {}}, \"additionalProperties\": true}}"));
        assertTrue(combined.contains("additionalProperties cannot be combined with patternProperties"), combined);
        String remote = errorOf(() -> resource.completions("{\"prompt\": \"x\", \"response_format\": {\"type\":"
                + " \"json_schema\", \"json_schema\": {\"name\": \"r\", \"schema\": {\"$ref\":"
                + " \"https://example.com/s.json\"}}}}"));
        assertTrue(remote.contains("unsupported $ref https://example.com/s.json")
                && remote.contains("remote schemas are not fetched"), remote);
        String bounds = errorOf(() -> resource.completions("{\"prompt\": \"x\", \"guided_json\": {\"type\": \"number\","
                + " \"minimum\": 2, \"maximum\": 1}}"));
        assertTrue(bounds.contains("no number lies within the minimum and maximum bounds"), bounds);
    }

    // -------------------------------------------------- structured_outputs JSON options

    /** The {@code structured_outputs} JSON constraint of a boolean object, with {@code options} appended. */
    private static String booleanObject(String options) {
        return "\"structured_outputs\": {\"json\": {\"type\": \"object\", \"properties\":"
                + " {\"a\": {\"type\": \"boolean\"}}, \"required\": [\"a\"]}" + options + "}";
    }

    @Test
    void disableAnyWhitespaceCompactsTheSampledJson() {
        String text = constrainedText(booleanObject(", \"disable_any_whitespace\": true"));
        assertTrue(text.matches("\\{\"a\":(true|false)\\}"), "the compact form carries no whitespace: " + text);
        assertTrue(Json.parse(text, Json.Limits.UNTRUSTED) instanceof Map);
    }

    @Test
    void whitespacePatternBoundsTheSyntacticWhitespace() {
        String text = constrainedText(booleanObject(", \"whitespace_pattern\": \"[ ]?\""));
        assertTrue(text.matches("\\{ ?\"a\" ?: ?(true|false) ?\\}"), "only spaces separate the tokens: " + text);
        assertFalse(text.contains("\n") || text.contains("\t") || text.contains("\r"), text);
    }

    @Test
    void disableAdditionalPropertiesClosesTheSampledObject() {
        Object value = constrained(booleanObject(", \"disable_additional_properties\": true"));
        assertTrue(value instanceof Map, "an object: " + value);
        Map<?, ?> object = (Map<?, ?>) value;
        assertEquals(java.util.Set.of("a"), object.keySet(), "only the declared property: " + value);
    }

    @Test
    void structuredOutputOptionsRefuseByName() {
        String invalidPattern = "{\"prompt\": \"x\", \"structured_outputs\": {\"json\": {\"type\": \"object\"},"
                + " \"whitespace_pattern\": \"\\\\b\"}}";
        assertEquals(400, statusOf(() -> resource.completions(invalidPattern)));
        assertTrue(errorOf(() -> resource.completions(invalidPattern)).contains("whitespace_pattern"));

        String wrongType = "{\"prompt\": \"x\", \"structured_outputs\": {\"json\": {\"type\": \"object\"},"
                + " \"disable_any_whitespace\": \"yes\"}}";
        assertEquals(400, statusOf(() -> resource.completions(wrongType)));
        assertTrue(errorOf(() -> resource.completions(wrongType)).contains("structured_outputs.disable_any_whitespace"));

        String nonJson = "{\"prompt\": \"x\", \"structured_outputs\": {\"regex\": \"[0-9]+\","
                + " \"disable_any_whitespace\": true}}";
        assertEquals(400, statusOf(() -> resource.completions(nonJson)));
        assertTrue(errorOf(() -> resource.completions(nonJson)).contains("disable_any_whitespace"));

        String noConstraint = "{\"prompt\": \"x\", \"structured_outputs\": {\"whitespace_pattern\": \"[ ]*\"}}";
        assertEquals(400, statusOf(() -> resource.completions(noConstraint)));
        assertTrue(errorOf(() -> resource.completions(noConstraint)).contains("structured_outputs needs a json"));
    }
}

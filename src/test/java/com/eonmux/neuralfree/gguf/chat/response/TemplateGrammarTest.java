package com.eonmux.neuralfree.gguf.chat.response;

import com.eonmux.neuralfree.gguf.chat.ChatTool;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.gguf.grammar.GbnfBuilder;
import com.eonmux.neuralfree.gguf.grammar.GrammarMatcher;
import com.eonmux.neuralfree.utils.json.Json;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The structural tags derived from a response template ({@link TemplateGrammar}, {@link TemplateArguments}), vLLM's
 * {@code tests/parser/test_response_template_grammar.py} case for case, each tag compiled by the grammar engine and
 * matched against whole outputs.
 */
public class TemplateGrammarTest {

    private static final String TEMPLATE = "{'start_anchor': '<assistant>', 'fields': {"
            + "'thinking': {'open': '<think>', 'close': '</think>', 'content': 'text'},"
            + "'content': {'content': 'text'},"
            + "'tool_calls': {'open_pattern': '<call name=\\'(?P<name>[\\\\w.-]+)\\'>', 'close': '</call>',"
            + " 'repeats': true, 'content': 'xml-inline',"
            + " 'content_args': {'tag_pattern': '<arg name=\\'(?P<key>[^\\']+)\\'>(?P<value>.*?)</arg>'},"
            + " 'transform': {'type': 'function', 'function': {'name': '{name}', 'arguments': '{content}'}}}}}";
    private static final String WEATHER_PARAMETERS = "{'type': 'object', 'properties': {'city': {'type': 'string'},"
            + " 'days': {'type': 'integer'}}, 'required': ['city']}";
    private static final ChatTool WEATHER = tool(WEATHER_PARAMETERS, "get.weather", true);
    private static final ChatTool PING = ChatTool.builder("ping").build();
    private static final String CALL = "<call name=\"get.weather\"><arg name=\"city\">Paris</arg></call>";

    /** JSON written with single quotes for readability; an escaped quote stays a double quote. */
    @SuppressWarnings("unchecked")
    static Map<String, Object> json(String singleQuoted) {
        return (Map<String, Object>) Json.parse(singleQuoted.replace("'", "\""));
    }

    private static Map<String, Object> template() {
        return json(TEMPLATE);
    }

    private static Map<String, Object> answerTemplate() {
        Map<String, Object> t = template();
        fields(t).put("content", json("{'open': '<answer>', 'close': '</answer>', 'content': 'text'}"));
        return t;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> fields(Map<String, Object> template) {
        return (Map<String, Object>) template.get("fields");
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> toolField(Map<String, Object> template) {
        return (Map<String, Object>) fields(template).get("tool_calls");
    }

    static ChatTool tool(String parameters, String name, Boolean strict) {
        ChatTool.Builder b = ChatTool.builder(name).strict(strict);
        return parameters == null ? b.build() : b.parameters(json(parameters)).build();
    }

    private static ChatTool tool(String parameters) {
        return tool(parameters, "f", true);
    }

    /** vLLM's {@code build}: the tag of {@code functions} over {@code template}. */
    private static final class Build {
        Map<String, Object> template = TemplateGrammarTest.template();
        List<ChatTool> functions = Arrays.asList(WEATHER, PING);
        boolean parallel = true;
        boolean thinking = true;
        boolean answer;
        Map<String, Object> answerFormat;
        boolean sortedArguments;
        List<String> messageStart;
        boolean requiresReasoning;

        Build template(Map<String, Object> v) { template = v; return this; }
        Build functions(ChatTool... v) { functions = Arrays.asList(v); return this; }

        Map<String, Object> tag() {
            ResponseTemplate t = ResponseTemplate.load(template);
            TemplateGrammar.CallOptions o = new TemplateGrammar.CallOptions();
            o.parallelToolCalls = parallel;
            o.thinking = thinking ? t.field(ResponseTemplate.THINKING) : null;
            o.answer = answer ? t.field(ResponseTemplate.CONTENT) : null;
            o.answerFormat = answerFormat;
            o.sortedArguments = sortedArguments;
            o.messageStart = messageStart;
            o.requiresReasoning = requiresReasoning;
            return TemplateGrammar.toolCallTag(t.field(ResponseTemplate.TOOL_CALLS), functions, o);
        }
    }

    static boolean accepts(Map<String, Object> tag, String text) {
        assertNotNull("the tag is derived", tag);
        String payload = Json.toString(tag, Json.Style.CANONICAL);
        return GrammarMatcher.matches(GbnfBuilder.grammarOf(ConstraintSpec.structuralTag(payload)), text);
    }

    private static String call(String... keyValues) {
        StringBuilder body = new StringBuilder();
        for (int i = 0; i < keyValues.length; i += 2) {
            body.append("<arg name=\"").append(keyValues[i]).append("\">").append(keyValues[i + 1]).append("</arg>");
        }
        return "<call name=\"f\">" + body + "</call>";
    }

    @Test
    public void forcedCallsFollowTheTemplateFormat() {
        Map<String, Object> tag = new Build().tag();
        assertTrue(accepts(tag, "<think>plan</think>" + CALL));
        assertTrue(accepts(tag, CALL));
        assertTrue(accepts(tag, CALL.substring(0, CALL.length() - "</call>".length())));
        assertTrue(accepts(tag, "<call name=\"get.weather\"><arg name=\"city\">Paris</arg>\n"
                + "<arg name=\"days\">3</arg></call>"));
        assertFalse(accepts(tag, "<call name=\"get.weather\"><arg name=\"city\">Paris</arg>"
                + "<arg name=\"days\">soon</arg></call>"));
        assertFalse(accepts(tag, "<call name=\"get.weather\"><arg name=\"days\">3</arg></call>"));
        assertTrue(accepts(tag, CALL + "<call name=\"ping\"><arg name=\"any\">x</arg></call>"));
        assertFalse(accepts(tag, "<call name=\"other\"></call>"));
        assertFalse(accepts(tag, "<think>plan</think>The answer."));
    }

    @Test
    public void oneCallWhenParallelCallsAreDisabled() {
        Build b = new Build();
        b.parallel = false;
        Map<String, Object> tag = b.tag();
        assertTrue(accepts(tag, CALL));
        assertFalse(accepts(tag, CALL + CALL));
    }

    @Test
    public void thinkingIsRejectedWhenNotRequested() {
        Build b = new Build();
        b.thinking = false;
        assertFalse(accepts(b.tag(), "<think>plan</think>" + CALL));
    }

    @Test
    public void anAnswerFieldLetsTheModelReplyInstead() {
        Build b = new Build().template(answerTemplate());
        b.answer = true;
        Map<String, Object> tag = b.tag();
        assertTrue(accepts(tag, "<think>plan</think><answer>Sunny."));
        assertTrue(accepts(tag, "<answer>Sunny.</answer>"));
        assertTrue(accepts(tag, "<think>plan</think>" + CALL));
        assertFalse(accepts(tag, "<answer>Sunny.</answer>" + CALL));
        assertFalse(accepts(tag, "<call name=\"get.weather\"><arg name=\"days\">soon</arg></call>"));
    }

    @Test
    public void requiredReasoningOpensTheMessageWithNonEmptyReasoning() {
        String[][] cases = {
                {"<think>plan</think>" + CALL, "false"},
                {"<think>\nplan</think><answer>Sunny.", "false"},
                {"<think> \n</think><answer>Sunny.", "false"},
                {CALL, "true"},
                {"<answer>Sunny.", "true"},
                {"<think></think><answer>Sunny.", "true"},
                {"<think><</think><answer>Sunny.", "true"},
        };
        for (boolean required : new boolean[]{false, true}) {
            Build b = new Build().template(answerTemplate()).functions(WEATHER);
            b.answer = true;
            b.messageStart = Collections.singletonList("<assistant>");
            b.requiresReasoning = required;
            Map<String, Object> tag = b.tag();
            for (String[] c : cases) {
                boolean optionalOnly = Boolean.parseBoolean(c[1]);
                assertEquals(required + " " + c[0], !required || !optionalOnly, accepts(tag, c[0]));
            }
        }
    }

    @Test
    public void theAnswerRestFinishesTheOpenAnswer() {
        ResponseTemplate.Field answer = ResponseTemplate.load(answerTemplate()).field(ResponseTemplate.CONTENT);
        Map<String, Object> tag = TemplateGrammar.answerRestTag(answer);
        assertTrue(accepts(tag, "Sunny.</answer>"));
        assertTrue(accepts(tag, "Sunny."));
        assertFalse(accepts(tag, "Sunny.</answer>" + CALL));
    }

    @Test
    public void anAnswerWithoutALiteralOpenerIsNotDerived() {
        Build b = new Build();
        b.answer = true;
        assertNull(b.tag());
    }

    @Test
    public void theAnswerHoldsTheAnswerFormat() {
        Map<String, Object> schema = json("{'type': 'json_schema', 'json_schema': {'type': 'object'}}");
        Build withTools = new Build().template(answerTemplate());
        withTools.answer = true;
        withTools.answerFormat = schema;
        Build answerOnly = new Build().template(answerTemplate()).functions();
        answerOnly.answer = true;
        answerOnly.answerFormat = schema;
        for (Map<String, Object> tag : Arrays.asList(withTools.tag(), answerOnly.tag())) {
            assertTrue(accepts(tag, "<think>plan</think><answer>{\"a\": 1}</answer>"));
            assertFalse(accepts(tag, "<answer>Sunny.</answer>"));
        }
        assertTrue(accepts(withTools.tag(), CALL));
        assertFalse(accepts(answerOnly.tag(), CALL));
    }

    @Test
    public void jsonArgumentsFollowTheSchema() {
        Map<String, Object> t = template();
        toolField(t).put("content", "json");
        toolField(t).put("content_args", new LinkedHashMap<String, Object>());
        Map<String, Object> tag = new Build().template(t).functions(WEATHER).tag();
        assertTrue(accepts(tag, "<call name=\"get.weather\">{\"city\": \"Paris\"}</call>"));
        assertFalse(accepts(tag, "<call name=\"get.weather\">{\"days\": 3}</call>"));
    }

    @Test
    public void underivableToolFormatsAreNotDerived() {
        String endX = "{'tag_pattern': '<arg name=\\'(?P<key>[^\\']+)\\'>(?P<value>.*?)X'}";
        String endXJson = "{'tag_pattern': '<arg name=\\'(?P<key>[^\\']+)\\'>(?P<value>.*?)X',"
                + " 'value_parser': {'name': 'json'}}";
        String[] changes = {
                "{'content': 'json', 'content_args': {'unquoted_keys': true}}",
                "{'transform': {'type': 'function', 'function': {'name': 'fixed', 'arguments': '{content}'}}}",
                "{'content': 'kv-lines', 'content_args': {}}",
                // framing in which argument text can spell a closer, where the parser would end the call
                "{'close': ['}X', '</call>'], 'content_args': " + endX + "}",
                "{'close': ['}X', '</call>'], 'content_args': " + endXJson + "}",
                "{'close': ['7X', '</call>'], 'content_args': " + endX + "}",
                "{'close': ['>X', '</call>']}",
                "{'close': ['></call>', '</call>']}",
                "{'close': ['><a', '</call>']}",
                "{'close': ['}</call>', '</call>'], 'content': 'json', 'content_args': {}}",
        };
        for (String change : changes) {
            Map<String, Object> t = template();
            toolField(t).putAll(json(change));
            assertNull(change, new Build().template(t).tag());
        }
    }

    @Test
    public void nonLiteralThinkingDelimitersAreNotDerived() {
        Map<String, Object> t = template();
        @SuppressWarnings("unchecked")
        Map<String, Object> thinking = (Map<String, Object>) fields(t).get("thinking");
        thinking.remove("open");
        thinking.put("open_pattern", "<think(?: mode=\\w+)?>");
        assertNull(new Build().template(t).tag());
        Build b = new Build().template(t);
        b.thinking = false;
        assertNotNull(b.tag());
    }

    @Test
    public void argumentValuesFollowHowTheParserReadsThem() {
        String level = "{'type': 'object', 'properties': {'level': {'$ref': '#/$defs/level'}}}";
        Object[][] cases = {
                {"{'type': 'integer', 'nullable': true}", "null", true},
                {"{'type': 'integer', 'nullable': true}", "soon", false},
                {"{'type': ['integer', 'null']}", "null", true},
                {"{'anyOf': [{'type': 'integer'}, {'type': 'boolean'}]}", "true", true},
                {"{'anyOf': [{'type': 'integer'}, {'type': 'boolean'}]}", "soon", false},
                {"{'type': ['string', 'integer']}", "soon", true},
                {"{'enum': ['celsius', 'fahrenheit']}", "celsius", true},
                {"{'enum': ['celsius', 'fahrenheit']}", "kelvin", false},
                {"{'type': 'string', 'const': 'on'}", "off", false},
                {level, "{\"level\": 2}", true},
                {level, "{\"level\": \"high\"}", false},
        };
        for (Object[] c : cases) {
            String parameters = "{'type': 'object', 'properties': {'v': " + c[0] + "}, 'required': ['v'],"
                    + " '$defs': {'level': {'type': 'integer'}}}";
            Map<String, Object> tag = new Build().functions(tool(parameters)).tag();
            assertEquals(c[0] + " " + c[1], c[2], accepts(tag, call("v", (String) c[1])));
        }
    }

    @Test
    public void schemasTextValuesCannotHoldAreNotDerived() {
        Object[][] cases = {
                // the parser reads values without a type as text
                {"{'enum': [1, 2]}", true, false},
                {"{'$ref': '#/$defs/level'}", true, false},
                {"{'$ref': '#/$defs/level'}", false, true},
                // raw text cannot hold its closer
                {"{'enum': ['a</arg>b']}", true, false},
                {"{'type': 'object', 'properties': {'x': {'$ref': '#/x'}}}", true, false},
                {"false", true, false},
                // a match could write the start of the value's end
                {"{'type': 'string', 'pattern': '^[a-z<]+$'}", true, false},
                {"{'type': 'string', 'pattern': '^[a-z<]+$'}", false, true},
                {"{'type': 'string', 'pattern': '^a.b$'}", true, false},
                // xgrammar holds length bounds only in JSON
                {"{'type': 'string', 'minLength': 3}", true, false},
                {"{'type': 'string', 'format': 'ipv6'}", true, false},
        };
        for (Object[] c : cases) {
            String parameters = "{'type': 'object', 'properties': {'v': " + c[0] + "},"
                    + " '$defs': {'level': {'type': 'integer'}}}";
            Map<String, Object> tag = new Build().functions(tool(parameters, "f", (Boolean) c[1])).tag();
            assertEquals(c[0] + " strict " + c[1], c[2], tag != null);
        }
    }

    @Test
    public void textValuesHoldStringFormatsAndPatterns() {
        Object[][] cases = {
                {"{'type': 'string', 'format': 'email'}", "a.b@x.co", true},
                {"{'type': 'string', 'format': 'email'}", "ab", false},
                {"{'type': 'string', 'format': 'date-time'}", "2026-10-06T12:00:00Z", true},
                {"{'type': 'string', 'format': 'date-time'}", "2026-13-06T12:00:00Z", false},
                {"{'type': 'string', 'pattern': '^[A-Z]{3}$'}", "ABC", true},
                {"{'type': 'string', 'pattern': '^[A-Z]{3}$'}", "ABCD", false},
                // like xgrammar, the format overrides the pattern and the length bounds
                {"{'type': 'string', 'format': 'uuid', 'pattern': '^x$', 'maxLength': 1}",
                        "123e4567-e89b-12d3-a456-426614174000", true},
        };
        for (Object[] c : cases) {
            String parameters = "{'type': 'object', 'properties': {'v': " + c[0] + "}, 'required': ['v']}";
            Map<String, Object> tag = new Build().functions(tool(parameters)).tag();
            assertEquals(c[0] + " " + c[1], c[2], accepts(tag, call("v", (String) c[1])));
        }
    }

    @Test
    public void valuesCannotHoldACloser() {
        Map<String, Object> t = template();
        toolField(t).put("close", new ArrayList<Object>(Arrays.asList("</call>", "STOP")));
        String text = "{'type': 'object', 'properties': {'v': {'type': 'string'}}}";
        String jsonText = "{'type': 'object', 'properties': {'v': {'type': 'array', 'items': {'type': 'string'}}}}";
        assertTrue(accepts(new Build().template(t).functions(tool(text)).tag(), call("v", "ab")));
        assertFalse(accepts(new Build().template(t).functions(tool(text)).tag(), call("v", "aSTOPb")));
        assertFalse(accepts(new Build().template(t).functions(tool(null, "f", false)).tag(), call("v", "aSTOPb")));
        assertTrue(accepts(new Build().template(t).functions(tool(jsonText)).tag(), call("v", "[\"ab\"]")));
        assertFalse(accepts(new Build().template(t).functions(tool(jsonText)).tag(), call("v", "[\"aSTOPb\"]")));
    }

    @Test
    public void extraArgumentsNeedAdditionalProperties() {
        String parameters = "{'type': 'object', 'properties': {'city': {'type': 'string'}}";
        String extra = call("city", "Paris", "zip", "75001");
        assertFalse(accepts(new Build().functions(tool(parameters + "}")).tag(), extra));
        // next to declared keys, an extra one could repeat them unchecked
        assertNull(new Build().functions(tool(parameters + ", 'additionalProperties': true}")).tag());
        String freeForm = "{'type': 'object', 'additionalProperties': true}";
        assertTrue(accepts(new Build().functions(tool(freeForm)).tag(), extra));
    }

    @Test
    public void onlyStrictToolsFollowTheirSchema() {
        String offSchema = call("days", "soon");
        assertTrue(accepts(new Build().functions(tool(WEATHER_PARAMETERS, "f", false)).tag(), offSchema));
        assertFalse(accepts(new Build().functions(tool(WEATHER_PARAMETERS)).tag(), offSchema));
        assertTrue(accepts(new Build().functions(tool(null)).tag(), call()));
        assertFalse(accepts(new Build().functions(tool(null)).tag(), call("a", "x")));
    }

    @Test
    public void annotationsKeepStrictArgumentsToTheirSchema() {
        String parameters = WEATHER_PARAMETERS.substring(0, WEATHER_PARAMETERS.length() - 1)
                + ", '$schema': 'https://json-schema.org/draft/2020-12/schema', '$id': 'weather'}";
        Map<String, Object> tag = new Build().functions(tool(parameters)).tag();
        assertTrue(accepts(tag, call("city", "Paris", "days", "3")));
        assertFalse(accepts(tag, call("city", "Paris", "days", "soon")));
    }

    @Test
    public void strictArgumentsFollowDeclaredOrSortedOrder() {
        String parameters = "{'properties': {'path': {'type': 'string'}, 'content': {'type': 'string'}},"
                + " 'required': ['path']}";
        Object[][] cases = {
                {new String[]{"path", "a", "content", "b"}, true, true},
                {new String[]{"content", "b", "path", "a"}, false, true},
                {new String[]{"path", "a"}, true, true},
                {new String[]{"content", "b"}, false, false},
                {new String[]{"path", "a", "path", "c"}, false, false},
                {new String[]{"path", "a", "mode", "c"}, false, false},
        };
        Build declared = new Build().functions(tool(parameters));
        Build sorted = new Build().functions(tool(parameters));
        sorted.sortedArguments = true;
        Map<String, Object> declaredTag = declared.tag();
        Map<String, Object> sortedTag = sorted.tag();
        for (Object[] c : cases) {
            String text = call((String[]) c[0]);
            assertEquals(text, c[1], accepts(declaredTag, text));
            assertEquals(text, c[2], accepts(sortedTag, text));
        }
    }

    private static Map<String, Object> lookaheadTemplate() {
        Map<String, Object> t = template();
        @SuppressWarnings("unchecked")
        Map<String, Object> args = (Map<String, Object>) toolField(t).get("content_args");
        args.put("tag_pattern", "<arg name=\"(?P<key>[^\"]+)\">(?P<value>.*?)\\n(?:</arg>|(?=<arg name=\"))");
        return t;
    }

    @Test
    public void valuesEndWhereALookaheadEndDoes() {
        Object[][] cases = {
                {"<arg name=\"city\">Paris\n</arg>\n<arg name=\"days\">3\n</arg>", true, true},
                {"<arg name=\"city\">Paris\n<arg name=\"days\">3\n</arg>", true, true},
                {"<arg name=\"days\">3\n<arg name=\"city\">Paris\n</arg>", true, false},
                {"<arg name=\"city\">Paris\n<arg name=\"city\">Lyon\n</arg>", true, false},
                {"<arg name=\"city\">Paris\n<arg name=\"hours\">3\n</arg>", true, false},
                {"<arg name=\"city\">Paris\n<arg name=\"", false, false},
                {"<arg name=\"city\">Paris\n<arg name=\"days\n</arg>", false, false},
                {"<arg name=\"city\">a <arg name=\"b\"> c\n</arg>", true, true},
        };
        Map<String, Object> loose = new Build().template(lookaheadTemplate())
                .functions(tool(WEATHER_PARAMETERS, "f", false)).tag();
        Map<String, Object> strict = new Build().template(lookaheadTemplate())
                .functions(tool(WEATHER_PARAMETERS, "f", true)).tag();
        for (Object[] c : cases) {
            String text = "<call name=\"f\">" + c[0] + "</call>";
            assertEquals("loose " + text, c[1], accepts(loose, text));
            assertEquals("strict " + text, c[2], accepts(strict, text));
        }
    }

    @Test
    public void theLastParameterFollowsEveryRequiredOne() {
        String parameters = "{'properties': {'a': {'type': 'string'}, 'b': {'type': 'string'},"
                + " 'c': {'type': 'string'}, 'd': {'type': 'string'}, 'e': {'type': 'string'}}, 'required': ['b']}";
        Map<String, Object> tag = new Build().template(lookaheadTemplate()).functions(tool(parameters)).tag();
        Object[][] cases = {
                {"b", true}, {"ab", true}, {"bc", true}, {"bd", true}, {"be", true}, {"abde", true},
                {"bcde", true}, {"", false}, {"a", false}, {"acd", false}, {"bdc", false}, {"bcc", false},
        };
        for (boolean lookahead : new boolean[]{false, true}) {
            for (Object[] c : cases) {
                String keys = (String) c[0];
                StringBuilder body = new StringBuilder();
                for (int i = 0; i < keys.length(); i++) {
                    boolean last = i == keys.length() - 1;
                    body.append("<arg name=\"").append(keys.charAt(i)).append("\">x")
                            .append(last || !lookahead ? "\n</arg>" : "\n");
                }
                String text = "<call name=\"f\">" + body + "</call>";
                assertEquals(lookahead + " " + keys, c[1], accepts(tag, text));
            }
        }
    }

    @Test
    public void aRequiredArgumentWithoutASchemaIsStillRequired() {
        String parameters = "{'type': 'object', 'properties': {}, 'required': ['a']}";
        Map<String, Object> tag = new Build().functions(tool(parameters)).tag();
        assertTrue(accepts(tag, call("a", "x")));
        assertFalse(accepts(tag, call()));
        assertNull(new Build().functions(tool("{'type': 'object', 'properties': {}, 'required': ['a', 1]}")).tag());
    }

    @Test
    public void unencodableNamesAreRejected() {
        rejects(new Build().functions(tool("{}", "get weather", true)), "tool name");
        rejects(new Build().functions(tool("{'properties': {'a\\'b': {'type': 'string'}}}")), "parameter name");
    }

    private static void rejects(Build b, String expected) {
        try {
            b.tag();
            fail("expected a refusal naming " + expected);
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains(expected));
        }
    }
}

package com.eonmux.neuralfree.gguf.chat.tools;

import com.eonmux.neuralfree.gguf.chat.ChatTool;
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

/**
 * The built-in tool-call formats: each reads the syntax its chat template writes into content and calls,
 * malformed blocks stay content, a final text cut inside a call keeps the finished calls, and every sample
 * streams — fed one character at a time and in uneven chunks — into deltas that concatenate to the final parse.
 */
public class ToolParsersTest {

    private static final String DS_CALLS = "<｜tool▁calls▁begin｜>";
    private static final String DS_CALLS_END = "<｜tool▁calls▁end｜>";
    private static final String DS_CALL = "<｜tool▁call▁begin｜>";
    private static final String DS_CALL_END = "<｜tool▁call▁end｜>";
    private static final String DS_SEP = "<｜tool▁sep｜>";

    /** get_weather(city: string, days: integer), the schema the XML formats type values by. */
    private static final List<ChatTool> TOOLS = Collections.singletonList(ChatTool.builder("get_weather")
            .parameters(schema()).build());

    private static Map<String, Object> schema() {
        Map<String, Object> city = new LinkedHashMap<>();
        city.put("type", "string");
        Map<String, Object> days = new LinkedHashMap<>();
        days.put("type", "integer");
        Map<String, Object> props = new LinkedHashMap<>();
        props.put("city", city);
        props.put("days", days);
        Map<String, Object> s = new LinkedHashMap<>();
        s.put("type", "object");
        s.put("properties", props);
        return s;
    }

    /** One sample: format, model text, expected content and calls (name, arguments JSON, id or null). */
    private static final class Sample {
        final String format;
        final String text;
        final String content;
        final String[][] calls;

        Sample(String format, String text, String content, String[]... calls) {
            this.format = format;
            this.text = text;
            this.content = content;
            this.calls = calls;
        }
    }

    private static List<Sample> samples() {
        return Arrays.asList(
                new Sample("hermes", "Let me check.\n\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": "
                        + "{\"city\": \"Paris\"}}\n</tool_call>", "Let me check.",
                        new String[]{"get_weather", "{\"city\": \"Paris\"}", null}),
                new Sample("hermes", "<tool_call>\n{\"name\": \"a\", \"arguments\": {}}\n</tool_call>\n<tool_call>\n"
                        + "{\"name\": \"b\", \"arguments\": {\"x\": [1, 2]}}\n</tool_call>", "",
                        new String[]{"a", "{}", null}, new String[]{"b", "{\"x\": [1, 2]}", null}),
                new Sample("hermes", "<tool_call>{\"name\": \"f\", \"arguments\": \"{\\\"a\\\": 1}\"}</tool_call>", "",
                        new String[]{"f", "{\"a\": 1}", null}),
                new Sample("hermes", "plain answer, no calls", "plain answer, no calls"),
                new Sample("ernie45", "<response>\nIt is sunny in Paris.</response>", "It is sunny in Paris."),
                new Sample("ernie45", "<tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\": 1}}\n</tool_call>\n", "",
                        new String[]{"f", "{\"a\": 1}", null}),
                new Sample("longcat", "Sure.<longcat_tool_call>{\"name\": \"f\", \"arguments\": {\"a\": 1}}"
                        + "</longcat_tool_call>", "Sure.", new String[]{"f", "{\"a\": 1}", null}),
                new Sample("qwen3_coder", "I'll look it up.\n\n<tool_call>\n<function=get_weather>\n"
                        + "<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n"
                        + "</tool_call>", "I'll look it up.",
                        new String[]{"get_weather", "{\"city\":\"Paris\",\"days\":3}", null}),
                new Sample("qwen3_coder", "<tool_call>\n<function=get_weather>\n<parameter=city>\nNew\nYork\n"
                        + "</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=noop>\n</function>\n"
                        + "</tool_call>", "",
                        new String[]{"get_weather", "{\"city\":\"New\\nYork\"}", null}, new String[]{"noop", "{}", null}),
                new Sample("qwen3_coder", "<function=get_weather>\n<parameter=days>\n2\n</parameter>\n</function>", "",
                        new String[]{"get_weather", "{\"days\":2}", null}),
                new Sample("seed_oss", "<seed:tool_call>\n<function=get_weather>\n<parameter=city>Rome</parameter>\n"
                        + "</function>\n</seed:tool_call>", "",
                        new String[]{"get_weather", "{\"city\":\"Rome\"}", null}),
                new Sample("minimax_m2", "Sure.\n<minimax:tool_call>\n<invoke name=\"get_weather\">\n"
                        + "<parameter name=\"city\">Oslo</parameter>\n<parameter name=\"days\">5</parameter>\n"
                        + "</invoke>\n<invoke name=\"noop\">\n</invoke>\n</minimax:tool_call>", "Sure.",
                        new String[]{"get_weather", "{\"city\":\"Oslo\",\"days\":5}", null},
                        new String[]{"noop", "{}", null}),
                new Sample("glm45", "<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>\n"
                        + "<arg_key>days</arg_key>\n<arg_value>1</arg_value>\n</tool_call>", "",
                        new String[]{"get_weather", "{\"city\":\"Paris\",\"days\":1}", null}),
                new Sample("glm47", "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Lima</arg_value>"
                        + "</tool_call>", "",
                        new String[]{"get_weather", "{\"city\":\"Lima\"}", null}),
                new Sample("llama3_json", "{\"name\": \"f\", \"parameters\": {\"a\": 1}}", "",
                        new String[]{"f", "{\"a\": 1}", null}),
                new Sample("llama3_json", "<|python_tag|>{\"name\": \"f\", \"parameters\": {}}; "
                        + "{\"name\": \"g\", \"parameters\": {\"b\": true}}", "",
                        new String[]{"f", "{}", null}, new String[]{"g", "{\"b\": true}", null}),
                new Sample("llama3_json", "Hello there", "Hello there"),
                new Sample("mistral", "[TOOL_CALLS][{\"name\": \"f\", \"arguments\": {\"a\": 1}}, "
                        + "{\"name\": \"g\", \"arguments\": {}}]", "",
                        new String[]{"f", "{\"a\": 1}", null}, new String[]{"g", "{}", null}),
                new Sample("mistral", "[TOOL_CALLS]f[ARGS]{\"a\": 1}[TOOL_CALLS]g[ARGS]{}", "",
                        new String[]{"f", "{\"a\": 1}", null}, new String[]{"g", "{}", null}),
                new Sample("mistral", "[TOOL_CALLS]f[CALL_ID]abc123def[ARGS]{\"q\": \"x\"}", "",
                        new String[]{"f", "{\"q\": \"x\"}", "abc123def"}),
                new Sample("deepseek_v3", "Checking." + DS_CALLS + DS_CALL + "function" + DS_SEP + "get_weather\n"
                        + "```json\n{\"city\": \"Paris\"}\n```" + DS_CALL_END + DS_CALLS_END, "Checking.",
                        new String[]{"get_weather", "{\"city\": \"Paris\"}", null}),
                new Sample("deepseek_v31", DS_CALLS + DS_CALL + "f" + DS_SEP + "{\"a\": 1}" + DS_CALL_END + DS_CALL
                        + "g" + DS_SEP + "{}" + DS_CALL_END + DS_CALLS_END, "",
                        new String[]{"f", "{\"a\": 1}", null}, new String[]{"g", "{}", null}),
                new Sample("kimi_k2", "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0"
                        + "<|tool_call_argument_begin|>{\"city\": \"Paris\"}<|tool_call_end|>"
                        + "<|tool_calls_section_end|>", "",
                        new String[]{"get_weather", "{\"city\": \"Paris\"}", "functions.get_weather:0"}),
                new Sample("internlm", "<|action_start|><|plugin|>\n{\"name\": \"f\", \"parameters\": {\"a\": 1}}"
                        + "<|action_end|>", "",
                        new String[]{"f", "{\"a\": 1}", null}),
                new Sample("jamba", "<tool_calls>[{\"name\": \"f\", \"arguments\": {}}]</tool_calls>", "",
                        new String[]{"f", "{}", null}),
                new Sample("granite", "<|tool_call|>[{\"name\": \"f\", \"arguments\": {\"a\": 1}}]", "",
                        new String[]{"f", "{\"a\": 1}", null}),
                new Sample("phi4_mini_json", "functools[{\"name\": \"f\", \"arguments\": {}}]", "",
                        new String[]{"f", "{}", null}),
                new Sample("xlam", "```json\n[{\"name\": \"f\", \"arguments\": {\"a\": 1}}]\n```", "",
                        new String[]{"f", "{\"a\": 1}", null}),
                new Sample("pythonic", "[get_weather(city='Paris', days=3), noop()]", "",
                        new String[]{"get_weather", "{\"city\":\"Paris\",\"days\":3}", null},
                        new String[]{"noop", "{}", null}),
                new Sample("llama4_pythonic", "<|python_start|>[f(a=True, b=None, c=[1, 2.5], d={'k': \"v\"})]"
                        + "<|python_end|>", "",
                        new String[]{"f", "{\"a\":true,\"b\":null,\"c\":[1,2.5],\"d\":{\"k\":\"v\"}}", null}),
                new Sample("lfm2", "<|tool_call_start|>[f(a=\"x\")]<|tool_call_end|>Done.", "Done.",
                        new String[]{"f", "{\"a\":\"x\"}", null}),
                new Sample("olmo3", "<function_calls>\nf(a=1)\ng(b='x')\n</function_calls>", "",
                        new String[]{"f", "{\"a\":1}", null}, new String[]{"g", "{\"b\":\"x\"}", null}));
    }

    private static ToolParser parser(String format) {
        ToolParser.Factory f = ToolParserRegistry.forName(format);
        assertNotNull("format " + format + " is registered", f);
        return f.create(TOOLS);
    }

    @Test
    public void everyFormatReadsItsTemplateSyntax() {
        for (Sample s : samples()) {
            ToolParse p = parser(s.format).parse(s.text, false);
            assertEquals(s.format + " content of " + s.text, s.content, p.content());
            assertEquals(s.format + " call count of " + s.text, s.calls.length, p.calls().size());
            for (int i = 0; i < s.calls.length; i++) {
                ToolCall c = p.calls().get(i);
                assertEquals(s.format + " name", s.calls[i][0], c.name());
                assertJsonEquals(s.format + " arguments of " + c.name(), s.calls[i][1], c.arguments());
                assertEquals(s.format + " id", s.calls[i][2], c.id());
                assertTrue(s.format + " call is complete", c.complete());
            }
        }
    }

    @Test
    public void everySampleStreamsIntoDeltasThatConcatenateToTheFinalParse() {
        for (Sample s : samples()) {
            for (int chunk : new int[]{1, 3, 7}) {
                Collected got = stream(parser(s.format), s.text, chunk);
                assertFalse(s.format + " stream diverged (chunk " + chunk + ") on " + s.text, got.stream.diverged());
                assertEquals(s.format + " streamed content (chunk " + chunk + ")", s.content, got.content.toString());
                assertEquals(s.format + " streamed call count", s.calls.length, got.names.size());
                for (int i = 0; i < s.calls.length; i++) {
                    assertEquals(s.format + " streamed name", s.calls[i][0], got.names.get(i));
                    assertJsonEquals(s.format + " streamed arguments (chunk " + chunk + ")", s.calls[i][1],
                            got.args.get(i).toString());
                    assertEquals("the streamed arguments are the final arguments",
                            got.stream.calls().get(i).arguments(), got.args.get(i).toString());
                }
            }
        }
    }

    @Test
    public void schemaTypesTheXmlValuesAndUndeclaredParametersStayText() {
        ToolParser untyped = ToolParserRegistry.forName("qwen3_coder").create(Collections.<ChatTool>emptyList());
        ToolParse p = untyped.parse("<tool_call>\n<function=get_weather>\n<parameter=days>\n3\n</parameter>\n"
                + "</function>\n</tool_call>", false);
        assertEquals("{\"days\":\"3\"}", p.calls().get(0).arguments());
        ToolParse typed = parser("qwen3_coder").parse("<tool_call>\n<function=get_weather>\n<parameter=days>\n"
                + "three\n</parameter>\n</function>\n</tool_call>", false);
        assertEquals("a value that is no JSON keeps its text", "{\"days\":\"three\"}",
                typed.calls().get(0).arguments());
    }

    @Test
    public void malformedBlocksStayContent() {
        String text = "Answer <tool_call>not json at all</tool_call> end";
        ToolParse p = parser("hermes").parse(text, false);
        assertEquals(0, p.calls().size());
        assertEquals(text, p.content());
    }

    @Test
    public void aFinalTextCutInsideACallKeepsTheFinishedCalls() {
        String text = "<tool_call>\n{\"name\": \"a\", \"arguments\": {}}\n</tool_call>\n<tool_call>\n"
                + "{\"name\": \"b\", \"argu";
        ToolParse p = parser("hermes").parse(text, false);
        assertEquals(1, p.calls().size());
        assertEquals("a", p.calls().get(0).name());
        assertTrue("the cut-off call stays visible as content: " + p.content(), p.content().contains("\"b\""));
        ToolParse unclosed = parser("hermes").parse("<tool_call>{\"name\": \"f\", \"arguments\": {\"x\": 1}}", false);
        assertEquals("a call the output ends inside of its closer stands", 1, unclosed.calls().size());
    }

    @Test
    public void partialParsesHoldBackOpenerPrefixesAndReportGrowingArguments() {
        ToolParser p = parser("hermes");
        assertEquals("Hi", p.parse("Hi <tool_", true).content());
        ToolParse mid = p.parse("<tool_call>{\"name\": \"f\", \"arguments\": {\"city\": \"Par", true);
        assertEquals(1, mid.calls().size());
        assertFalse(mid.calls().get(0).complete());
        assertEquals("{\"city\": \"Par", mid.calls().get(0).arguments());
        ToolParse xml = parser("qwen3_coder").parse("<tool_call>\n<function=get_weather>\n<parameter=city>\nPar",
                true);
        assertEquals("an open parameter is not reported yet", "{", xml.calls().get(0).arguments());
    }

    @Test
    public void templatesSelectTheirFormat() {
        assertEquals("qwen3_coder", detect("{{ '<tool_call>\\n<function=' + name + '>\\n<parameter=' }}"));
        assertEquals("hermes", detect("<tool_call>\\n{\"name\": ...}\\n</tool_call>"));
        assertEquals("longcat", detect("<longcat_tool_call>{{ call }}</longcat_tool_call>"));
        assertEquals("ernie45", detect("<tool_call>{{ call }}</tool_call><response>{{ content }}</response>"));
        assertEquals("glm45", detect("<tool_call>{{ name }}<arg_key>k</arg_key>"));
        assertEquals("minimax_m2", detect("<minimax:tool_call><invoke name=\"x\">"));
        assertEquals("seed_oss", detect("<seed:tool_call><function="));
        assertEquals("mistral", detect("[TOOL_CALLS]"));
        assertEquals("kimi_k2", detect("<|tool_calls_section_begin|><|tool_call_begin|>"));
        assertEquals("deepseek_v31", detect(DS_CALL + "name" + DS_SEP));
        assertEquals("deepseek_v3", detect(DS_CALL + "function" + DS_SEP + "name\n```json"));
        assertEquals("llama3_json", detect("<|python_tag|>"));
        assertEquals("lfm2", detect("<|tool_call_start|>["));
        assertEquals("granite", detect("<|tool_call|>"));
        assertEquals("jamba", detect("<tool_calls>["));
        assertNull(ToolParserRegistry.detect("{{ messages }}"));
    }

    @Test
    public void aliasesAndSpellingsSelectTheSameFormat() {
        assertEquals(ToolParserRegistry.forName("qwen3_coder"), ToolParserRegistry.forName("Qwen3-XML"));
        assertEquals(ToolParserRegistry.forName("glm45"), ToolParserRegistry.forName("glm47"));
        assertTrue(ToolParserRegistry.names().contains("hermes"));
        assertNull(ToolParserRegistry.forName("no-such-parser"));
    }

    @Test
    public void aPluginNameCollidingWithABuiltInIsRefused() {
        ToolParser.Factory clash = new ToolParser.Factory() {
            @Override public String name() { return "plugin_fmt"; }
            @Override public List<String> aliases() { return Collections.singletonList("hermes"); }
            @Override public ToolParser create(List<ChatTool> tools) { return null; }
        };
        try {
            ToolParserRegistry.build(Collections.singletonList(ToolParserRegistry.forName("hermes")),
                    Collections.singletonList(() -> Collections.singletonList(clash)));
            throw new AssertionError("expected the collision to be refused");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("hermes") && e.getMessage().contains("built-in"));
        }
    }

    // ------------------------------------------------------------------ helpers

    private static String detect(String template) {
        ToolParser.Factory f = ToolParserRegistry.detect(template);
        return f == null ? null : f.name();
    }

    private static final class Collected implements ToolCallStream.Listener {
        final StringBuilder content = new StringBuilder();
        final List<String> names = new ArrayList<>();
        final List<StringBuilder> args = new ArrayList<>();
        ToolCallStream stream;

        @Override public void content(String delta) {
            content.append(delta);
        }

        @Override public void callStarted(int index, String id, String name) {
            assertEquals("calls start in order", names.size(), index);
            assertNotNull(id);
            names.add(name);
            args.add(new StringBuilder());
        }

        @Override public void arguments(int index, String delta) {
            args.get(index).append(delta);
        }
    }

    private static Collected stream(ToolParser parser, String text, int chunk) {
        Collected c = new Collected();
        ToolCallStream s = new ToolCallStream(parser);
        c.stream = s;
        for (int i = 0; i < text.length(); i += chunk) {
            s.feed(text.substring(i, Math.min(text.length(), i + chunk)), c);
        }
        s.finish(c);
        return c;
    }

    private static void assertJsonEquals(String what, String expected, String actual) {
        assertEquals(what, Json.parse(expected), Json.parse(actual));
    }
}

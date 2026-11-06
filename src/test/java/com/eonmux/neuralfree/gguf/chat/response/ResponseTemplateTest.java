package com.eonmux.neuralfree.gguf.chat.response;

import com.eonmux.neuralfree.gguf.chat.ChatTool;
import org.junit.Test;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The {@code response_template} loader and parser against vLLM's {@code chat_parsing} behavior: the loader's
 * validation by name, Python regex semantics in Java (named groups, partial matches), the literal expansion of
 * delimiter patterns, and whole-message parsing of the Gemma 4 checkpoint template — regions, the JSON dialect knobs,
 * transforms with captures, and tool arguments typed by the tools' schemas.
 */
public class ResponseTemplateTest {

    static final String PREFIX = "<|turn>model\n";
    static final String THINKING = "<|channel>thought\nplan<channel|>";
    static final String TOOL_GENERATION = "<|tool_call>call:set_alarm{hour:7,label:<|\"|>morning<|\"|>}<tool_call|>";

    /** The Gemma 4 instruction checkpoints' public metadata (vLLM tests/parser/test_response_template.py). */
    static ResponseTemplate gemma4() {
        try (InputStream in = ResponseTemplateTest.class.getResourceAsStream("/response-templates/gemma4.json")) {
            return ResponseTemplate.parse(new String(in.readAllBytes(), StandardCharsets.UTF_8));
        } catch (IOException e) {
            throw new AssertionError(e);
        }
    }

    static String gemma4Json() {
        try (InputStream in = ResponseTemplateTest.class.getResourceAsStream("/response-templates/gemma4.json")) {
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new AssertionError(e);
        }
    }

    private static Map<String, Object> parseMessage(ResponseTemplate t, String prefix, String text, ToolArguments tools) {
        ResponseParser p = new ResponseParser(t, prefix, tools, null);
        p.feed(text, new ResponseParser.Listener() {
        });
        return p.finish(new ResponseParser.Listener() {
        });
    }

    @Test
    public void aGemma4MessageParsesIntoItsRegions() {
        Map<String, Object> m = parseMessage(gemma4(), PREFIX, THINKING + "The answer is 42.<turn|>", ToolArguments.NONE);
        Map<String, Object> expected = new LinkedHashMap<>();
        expected.put("role", "assistant");
        expected.put("thinking", "plan");
        expected.put("content", "The answer is 42.");
        assertEquals(expected, m);
    }

    @Test
    @SuppressWarnings("unchecked")
    public void aToolRegionParsesThroughTheJsonDialectAndTheTransform() {
        Map<String, Object> m = parseMessage(gemma4(), PREFIX, "Let me check." + TOOL_GENERATION, ToolArguments.NONE);
        assertEquals("Let me check.", m.get("content"));
        List<Object> calls = (List<Object>) m.get("tool_calls");
        assertEquals(1, calls.size());
        Map<String, Object> call = (Map<String, Object>) calls.get(0);
        assertEquals("function", call.get("type"));
        Map<String, Object> fn = (Map<String, Object>) call.get("function");
        assertEquals("set_alarm", fn.get("name"));
        Map<String, Object> args = new LinkedHashMap<>();
        args.put("hour", 7L);
        args.put("label", "morning");
        assertEquals(args, fn.get("arguments"));
    }

    @Test
    @SuppressWarnings("unchecked")
    public void textArgumentsTakeTheTypesOfTheToolSchemas() {
        String json = "{\"start_anchor\": \"<a>\", \"fields\": {\"tool_calls\": {\"open_pattern\": \"<call (?P<name>\\\\w+)>\","
                + " \"close\": \"</call>\", \"content\": \"kv-lines\", \"transform\": {\"function\":"
                + " {\"name\": \"{name}\", \"arguments\": \"{content}\"}}}, \"content\": {}}}";
        Map<String, Object> props = new LinkedHashMap<>();
        props.put("count", Collections.singletonMap("type", "integer"));
        props.put("score", Collections.singletonMap("type", "number"));
        props.put("on", Collections.singletonMap("type", "boolean"));
        props.put("tags", Collections.singletonMap("type", "array"));
        props.put("note", Collections.singletonMap("anyOf", Arrays.asList(
                Collections.singletonMap("type", "integer"), Collections.singletonMap("type", "null"))));
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("type", "object");
        params.put("properties", props);
        ToolArguments tools = ToolArguments.of(Collections.singletonList(
                ChatTool.builder("set").parameters(params).build()));
        Map<String, Object> m = parseMessage(ResponseTemplate.parse(json), "",
                "<call set>count: 3.0\nscore: 2\non: True\ntags: [\"a\"]\nnote: None\nother: 5</call>", tools);
        Map<String, Object> args = (Map<String, Object>) ((Map<String, Object>) m.get("tool_calls")).get("function");
        args = (Map<String, Object>) args.get("arguments");
        assertEquals(3L, args.get("count"));
        assertEquals(2L, args.get("score"));
        assertEquals(Boolean.TRUE, args.get("on"));
        assertEquals(Collections.singletonList("a"), args.get("tags"));
        assertTrue(args.containsKey("note"));
        assertNull(args.get("note"));
        assertEquals("an argument without a schema keeps its text", "5", args.get("other"));
    }

    @Test
    public void xmlInlineAndScalarParsersReadTheirBodies() {
        String json = "{\"start_anchor\": \"<a>\", \"fields\": {"
                + "\"args\": {\"open\": \"<args>\", \"close\": \"</args>\", \"content\": \"xml-inline\", \"content_args\":"
                + " {\"tag_pattern\": \"<(?P<key>\\\\w+)>(?P<value>.*?)</(?P=key)>\", \"merge_duplicates\": true,"
                + " \"value_parser\": {\"name\": \"int\"}}},"
                + "\"n\": {\"open\": \"<n>\", \"close\": \"</n>\", \"content\": \"float\"},"
                + "\"ok\": {\"open\": \"<ok>\", \"close\": \"</ok>\", \"content\": \"bool\"}}}";
        Map<String, Object> m = parseMessage(ResponseTemplate.parse(json), "",
                "<args><x>1</x><x>2</x><y> 3 </y></args><n> 1_000.5 </n><ok>TRUE</ok>", ToolArguments.NONE);
        Map<String, Object> args = new LinkedHashMap<>();
        args.put("x", Arrays.asList(1L, 2L));
        args.put("y", 3L);
        assertEquals(args, m.get("args"));
        assertEquals(1000.5, m.get("n"));
        assertEquals(Boolean.TRUE, m.get("ok"));
    }

    @Test
    public void aBodyThatDoesNotParseEndsItsRegionAsMalformedAndParsingGoesOn() {
        ResponseParser p = new ResponseParser(gemma4(), PREFIX, ToolArguments.NONE, null);
        StringBuilder events = new StringBuilder();
        ResponseParser.Listener l = new ResponseParser.Listener() {
            @Override public void malformed(String field, int start, int end, String error) {
                events.append("malformed ").append(field);
            }
        };
        p.feed("<|tool_call>call:f{broken<tool_call|>after", l);
        Map<String, Object> m = p.finish(l);
        assertEquals("malformed tool_calls", events.toString());
        assertEquals("after", m.get("content"));
        assertFalse(m.containsKey("tool_calls"));
    }

    @Test
    public void thePromptOpensTheRegionItLeavesOpenAndItsTextNeverStreams() {
        StringBuilder chunks = new StringBuilder();
        ResponseParser.Listener l = new ResponseParser.Listener() {
            @Override public void chunk(String field, String text, boolean raw) {
                chunks.append(field).append(':').append(text).append('|');
            }
        };
        ResponseParser p = new ResponseParser(gemma4(), "<|turn>user\nhi" + PREFIX + "<|channel>thought\n",
                ToolArguments.NONE, l);
        assertEquals("thinking", p.currentField());
        assertEquals("", chunks.toString());
        p.feed("more<channel|>Answer<tu", l);
        assertEquals("the closer's possible start waits", "thinking:more|content:Answer|", chunks.toString());
        assertTrue(p.holding());
        p.feed("rn|>", l);
        assertEquals("thinking:more|content:Answer|", chunks.toString());
        assertEquals("Answer", p.finish(l).get("content"));
    }

    @Test
    public void theLoaderRefusesInvalidTemplatesByName() {
        refuses("{\"fields\": {\"content\": {}}}", "start_anchor");
        refuses("{\"start_anchor\": \"a\", \"fields\": {}}", "fields must be a non-empty dict");
        refuses("{\"start_anchor\": \"a\", \"version\": 2, \"fields\": {\"c\": {}}}", "version: 2");
        refuses("{\"start_anchor\": \"a\", \"extra\": 1, \"fields\": {\"c\": {}}}", "Unknown keys in response_template: [extra]");
        refuses("{\"start_anchor\": \"a\", \"fields\": {\"a\": {}, \"b\": {}}}", "At most one field");
        refuses("{\"start_anchor\": \"a\", \"fields\": {\"a\": {\"join\": \",\"}}}", "'join' requires 'repeats'");
        refuses("{\"start_anchor\": \"a\", \"fields\": {\"a\": {\"content\": \"yaml\"}}}", "unknown content parser 'yaml'");
        refuses("{\"start_anchor\": \"a\", \"fields\": {\"a\": {\"open\": \"x\", \"open_pattern\": \"x\"}}}",
                "cannot specify both 'open' and 'open_pattern'");
        refuses("{\"start_anchor\": \"a\", \"fields\": {\"a\": {\"open_pattern\": \"(?P<n>x)\"}}}", "named group(s) [n]");
        refuses("{\"start_anchor\": \"a\", \"fields\": {\"a\": {\"transform\": {\"x\": \"pre {content}\"}}}}",
                "mixes a {placeholder} with literal text");
        refuses("{\"start_anchor\": \"a\", \"fields\": {\"a\": {\"open_pattern\": \"(unclosed\"}}}", "invalid regex");
        refuses("{\"start_anchor\": \"a\", \"fields\": {\"a\": {\"content\": \"xml-inline\"}}}", "'tag_pattern'");
    }

    @Test
    public void theLoaderRefusesMalformedFieldPiecesByName() {
        String head = "{\"start_anchor\": \"a\", \"fields\": {\"t\": ";
        refuses(head + "{\"open\": \"\"}}}", "'open' literals cannot be empty strings");
        refuses(head + "{\"open\": []}}}", "'open' list must contain at least one literal");
        refuses(head + "{\"open\": [\"<t>\", 3]}}}", "'open' list must contain only strings");
        refuses(head + "{\"open_pattern\": 3}}}", "'open_pattern' must be a string");
        refuses(head + "{\"open\": \"<t>\", \"transform_each\": true}}}",
                "transform_each is set but no transform was provided");
        refuses(head + "{\"open\": \"<t>\", \"transform\": \"{content}\", \"transform_each\": \"yes\"}}}",
                "transform_each must be a bool, got str");
        refuses(head + "\"nope\"}}", "Field 't' must be a dict");
        refuses(head + "{\"open\": \"<t>\", \"bogus\": 1}}}", "Field 't': unknown keys [bogus]");
        refuses(head + "{\"close_pattern\": \"([unclosed\"}}}", "Field 't': close_pattern: invalid regex");
        refuses("{\"start_anchor\": \"a\", \"defaults\": 3, \"fields\": {\"c\": {}}}", "defaults must be a dict");
        refuses("\"nope\"", "response_template must be a dict, got str");
        ResponseTemplate eachOff = ResponseTemplate.parse(head + "{\"open\": \"<t>\", \"transform\": \"{content}\","
                + " \"transform_each\": false}}}");
        assertFalse(eachOff.field("t").transformEach());
    }

    @Test
    @SuppressWarnings("unchecked")
    public void aLoadedTemplateIsImmutableAndIndependentOfItsSpec() {
        Map<String, Object> spec = (Map<String, Object>) com.eonmux.neuralfree.utils.json.Json.parse(gemma4Json());
        ResponseTemplate t = ResponseTemplate.load(spec);
        ((Map<String, Object>) spec.get("fields")).put("thinking", Collections.singletonMap("open", "<think>"));
        ((Map<String, Object>) spec.get("defaults")).put("role", "changed");
        assertEquals(Collections.singletonList("<|channel>thought\n"), t.field("thinking").openLiterals());
        assertEquals("assistant", t.defaults().get("role"));
        try {
            t.defaults().put("role", "changed");
            fail("the defaults are unmodifiable");
        } catch (UnsupportedOperationException expected) {
            assertEquals("assistant", t.defaults().get("role"));
        }
    }

    @Test
    public void thePromptTruncatesPastItsLastStartAnchor() {
        ResponseTemplate t = gemma4();
        String prompt = "<bos><|turn>user\nhi<turn|>\n" + PREFIX;
        assertEquals("", t.truncatePastLastAnchor(prompt));
        assertEquals("<|channel>thought\n", t.truncatePastLastAnchor(prompt + "<|channel>thought\n"));
        assertEquals("the later anchor wins", "out", t.truncatePastLastAnchor(prompt + "x<tool_response|>out"));
        assertEquals("<|turn>user\nhi", t.truncatePastLastAnchor("<|turn>user\nhi"));
    }

    private static void refuses(String json, String message) {
        try {
            ResponseTemplate.parse(json);
            fail("expected a refusal naming " + message);
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains(message));
        }
    }

    @Test
    public void servingKeepsTheThreeRoutableFieldsAndAddsAnImplicitContent() {
        ResponseTemplate gemma = gemma4().forServing();
        assertEquals("content", gemma.implicitField());
        ResponseTemplate explicit = ResponseTemplate.parse("{\"start_anchor\": \"a\", \"fields\": {\"thinking\":"
                + " {\"open\": \"<t>\", \"close\": \"</t>\", \"optional\": false}}}").forServing();
        assertEquals("content", explicit.implicitField());
        assertFalse(explicit.field("content").opens());
        servingRefuses("{\"start_anchor\": \"a\", \"fields\": {\"summary\": {}}}", "unsupported semantic fields: [summary]");
        servingRefuses("{\"start_anchor\": \"a\", \"defaults\": {\"name\": \"x\"}, \"fields\": {\"content\": {}}}",
                "unsupported semantic fields: [name]");
        servingRefuses("{\"start_anchor\": \"a\", \"fields\": {\"thinking\": {\"open\": \"<t>\", \"content\": \"json\"}}}",
                "field 'thinking' uses semantics that cannot be streamed");
        servingRefuses("{\"start_anchor\": \"a\", \"fields\": {\"thinking\": {\"open\": \"<t>\", \"transform\":"
                + " \"{content}\"}}}", "field 'thinking' uses semantics that cannot be streamed");
        servingRefuses("{\"start_anchor\": \"a\", \"fields\": {\"content\": {\"open\": \"<c>\", \"repeats\": true,"
                + " \"join\": \"\"}}}", "field 'content' uses semantics that cannot be streamed");
    }

    @Test
    public void servingTurnsARequiredFieldOptional() {
        // vLLM test_delimited_template_keeps_plain_content_and_tolerates_required_fields
        String delimited = "{\"start_anchor\": \"<assistant>\", \"fields\": {"
                + "\"thinking\": {\"open\": \"<think>\", \"close\": \"</think>\"},"
                + "\"tool_calls\": {\"open\": \"<call>\", \"close\": \"</call>\", \"content\": \"json\","
                + " \"optional\": false}}}";
        assertNull(ResponseTemplate.parse(delimited).implicitField());
        try {
            parseMessage(ResponseTemplate.parse(delimited), "<assistant>", "<think>plan</think>answer",
                    ToolArguments.NONE);
            fail("the loaded template requires its tool_calls field");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("[tool_calls]"));
        }
        Map<String, Object> m = parseMessage(ResponseTemplate.parse(delimited).forServing(), "<assistant>",
                "<think>plan</think>answer", ToolArguments.NONE);
        assertEquals("plan", m.get("thinking"));
        assertEquals("answer", m.get("content"));
    }

    private static void servingRefuses(String json, String message) {
        try {
            ResponseTemplate.parse(json).forServing();
            fail("expected a refusal naming " + message);
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains(message));
        }
    }

    @Test
    public void pythonPatternsKeepTheirSemanticsInJava() {
        TemplateRegex named = TemplateRegex.compile("<(?P<tag_name>\\w+)>(?P<body>.*?)</(?P=tag_name)>\\Z", "test");
        TemplateRegex.Match m = named.search("<é>x\ny</é>", 0, false);
        assertEquals("é", m.captures.get("tag_name"));
        assertEquals("x\ny", m.captures.get("body"));
        assertNull("\\Z is the absolute end", named.search("<a>x</a>\n", 0, false));
        assertEquals("a literal { and the {,n} quantifier", 0,
                TemplateRegex.compile("{x}a{,2}[[]", "test").search("{x}aa[", 0, false).start);
        TemplateRegex.Match partial = TemplateRegex.compile("<\\|tool_call>call:\\w+", "test")
                .search("text <|tool_", 0, true);
        assertTrue(partial.partial);
        assertEquals(5, partial.start);
        assertNull(TemplateRegex.compile("abc", "test").search("xyz", 0, true).captures.get("none"));
    }

    @Test
    public void delimiterPatternsSpellTheirLiteralsWhenTheyHaveNoOtherSyntax() {
        assertEquals(Collections.singletonList("<|channel>final\n"), LiteralPatterns.expand("<\\|channel>final\\n"));
        assertEquals(Arrays.asList("a", "ab", "ac"), LiteralPatterns.expand("a(?:b|c)?"));
        assertNull(LiteralPatterns.expand("a\\w+"));
        assertNull(LiteralPatterns.expand("^a"));
    }
}

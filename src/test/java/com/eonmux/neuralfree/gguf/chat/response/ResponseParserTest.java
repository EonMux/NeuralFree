package com.eonmux.neuralfree.gguf.chat.response;

import com.eonmux.neuralfree.gguf.chat.ChatTool;
import com.eonmux.neuralfree.utils.json.Json;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.gguf.chat.response.ResponseTemplateTest.gemma4;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The streaming {@link ResponseParser} against the behavior vLLM's {@code ResponseParser} shows for the same templates:
 * streamed chunks equal a whole parse at every chunk size, a pending literal or regex delimiter holds the text from its
 * start (a regex opener paused mid-match included, through partial matching), region events carry the opener's span and
 * captures, a body that fails to parse or join ends its region malformed, and the parsed message keeps the defaults and
 * drops empty fields. Also the Python pattern syntax templates use, read with Python's meaning.
 */
public class ResponseParserTest {

    private static final String THINKING = "<|channel>thought\nplan<channel|>";
    private static final String ALARM = "<|tool_call>call:set_alarm{hour:7,label:<|\"|>morning<|\"|>}<tool_call|>";
    private static final String WEATHER = "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>";

    /** Records the events of a parse. */
    private static final class Recorder implements ResponseParser.Listener {
        final Map<String, StringBuilder> chunks = new LinkedHashMap<>();
        final List<String> events = new ArrayList<>();
        final Map<String, Map<String, String>> captures = new LinkedHashMap<>();
        final Map<String, int[]> openSpans = new LinkedHashMap<>();

        @Override public void opened(String field, int start, int end, Map<String, String> c) {
            events.add("open " + field);
            captures.put(field, c);
            openSpans.put(field, new int[]{start, end});
        }

        @Override public void chunk(String field, String text, boolean raw) {
            chunks.computeIfAbsent(field, k -> new StringBuilder()).append(text);
        }

        @Override public void closed(String field, int start, int end, Object value) {
            events.add("close " + field);
        }

        @Override public void malformed(String field, int start, int end, String error) {
            events.add("malformed " + field + ": " + error);
        }

        String text(String field) {
            StringBuilder b = chunks.get(field);
            return b == null ? "" : b.toString();
        }
    }

    private static Map<String, Object> parse(ResponseTemplate t, String text) {
        return parse(t, text, ToolArguments.NONE, new Recorder());
    }

    private static Map<String, Object> parse(ResponseTemplate t, String text, ToolArguments tools, Recorder r) {
        ResponseParser p = new ResponseParser(t, "", tools, null);
        p.feed(text, r);
        return p.finish(r);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> function(Object call) {
        return (Map<String, Object>) ((Map<String, Object>) call).get("function");
    }

    @Test
    public void aContentOnlyMessageKeepsTheDefaultsAndDropsTheEmptyFields() {
        Map<String, Object> message = parse(gemma4(), "Hello<turn|>");
        Map<String, Object> expected = new LinkedHashMap<>();
        expected.put("role", "assistant");
        expected.put("content", "Hello");
        assertEquals(expected, message);
    }

    @Test
    @SuppressWarnings("unchecked")
    public void repeatedToolRegionsCollectInOrderWithTheirDecodedKinds() {
        Map<String, Object> message = parse(gemma4(), WEATHER + ALARM
                + "<|tool_call>call:configure{enabled:false,ratio:5.0,label:null}<tool_call|>");
        List<Object> calls = (List<Object>) message.get("tool_calls");
        assertEquals(3, calls.size());
        assertEquals("get_weather", function(calls.get(0)).get("name"));
        assertEquals("set_alarm", function(calls.get(1)).get("name"));
        Map<String, Object> args = (Map<String, Object>) function(calls.get(2)).get("arguments");
        assertEquals(Boolean.FALSE, args.get("enabled"));
        assertEquals(5.0, args.get("ratio"));
        assertTrue(args.containsKey("label"));
        assertNull(args.get("label"));
    }

    @Test
    @SuppressWarnings("unchecked")
    public void aMalformedToolRegionIsDroppedAndTheNextOneParses() {
        Recorder r = new Recorder();
        Map<String, Object> message = parse(gemma4(), "<|tool_call>call:set_alarm{broken<tool_call|>" + ALARM,
                ToolArguments.NONE, r);
        assertTrue(r.events.toString(), r.events.get(1).startsWith("malformed tool_calls: json parser could not"));
        List<Object> calls = (List<Object>) message.get("tool_calls");
        assertEquals(1, calls.size());
        assertEquals("set_alarm", function(calls.get(0)).get("name"));
        assertEquals(Arrays.asList("role", "tool_calls"), new ArrayList<>(message.keySet()));
    }

    @Test
    public void streamingAtEveryChunkSizeMatchesTheWholeParse() {
        String[] texts = {THINKING + "The answer is 42.<turn|>", THINKING + ALARM, "Let me check. " + ALARM,
                WEATHER + ALARM, "Hi <|tool_cal not a call<turn|>"};
        for (String text : texts) {
            Recorder whole = new Recorder();
            Map<String, Object> expected = parse(gemma4(), text, ToolArguments.NONE, whole);
            for (int chunk : new int[]{1, 2, 7, 10_000}) {
                Recorder streamed = new Recorder();
                ResponseParser p = new ResponseParser(gemma4(), "", ToolArguments.NONE, null);
                for (int i = 0; i < text.length(); i += chunk) {
                    p.feed(text.substring(i, Math.min(i + chunk, text.length())), streamed);
                }
                Map<String, Object> message = p.finish(streamed);
                String at = "chunk " + chunk + " of " + text;
                assertEquals(at, whole.text("thinking"), streamed.text("thinking"));
                assertEquals(at, whole.text("content"), streamed.text("content"));
                assertEquals(at, whole.events, streamed.events);
                assertEquals(at, expected, message);
            }
        }
    }

    @Test
    public void aPendingLiteralCloserHoldsTheStreamBack() {
        Recorder r = new Recorder();
        ResponseParser p = new ResponseParser(gemma4(), "", ToolArguments.NONE, null);
        p.feed("Hello<turn", r);
        assertEquals("only the text before a possible <turn|> is final", "Hello", r.text("content"));
        assertTrue(p.holding());
        p.feed("|>", r);
        p.feed("", r);
        assertEquals("Hello", r.text("content"));
    }

    @Test
    @SuppressWarnings("unchecked")
    public void aRegexOpenerPausedMidMatchIsNeverReleasedAsContent() {
        // the variable part sits inside the opener, so a stream that stops in the name is an unfinished match
        ResponseTemplate t = ResponseTemplate.parse("{\"start_anchor\": \"<a>\", \"fields\": {\"content\": {},"
                + " \"tool_calls\": {\"open_pattern\": \"<call name=\\\"(?P<tool_name>\\\\w+)\\\">\","
                + " \"close\": \"</call>\", \"content\": \"json\","
                + " \"transform\": {\"name\": \"{tool_name}\", \"arguments\": \"{content}\"}}}}");
        String text = "Hi <call name=\"get_weather\">{\"city\": \"Paris\"}</call> bye";
        for (int chunk = 1; chunk <= text.length(); chunk++) {
            Recorder r = new Recorder();
            ResponseParser p = new ResponseParser(t, "", ToolArguments.NONE, null);
            for (int i = 0; i < text.length(); i += chunk) {
                p.feed(text.substring(i, Math.min(i + chunk, text.length())), r);
            }
            Map<String, Object> message = p.finish(r);
            assertEquals("chunk " + chunk, "Hi  bye", r.text("content"));
            Map<String, Object> call = (Map<String, Object>) message.get("tool_calls");
            assertEquals("chunk " + chunk, "get_weather", call.get("name"));
            assertEquals("chunk " + chunk, "Paris", ((Map<String, Object>) call.get("arguments")).get("city"));
        }
        Recorder paused = new Recorder();
        ResponseParser p = new ResponseParser(t, "", ToolArguments.NONE, null);
        p.feed("Hi <call name=\"get_wea", paused);
        assertEquals("Hi ", paused.text("content"));
        assertTrue(p.holding());
    }

    @Test
    public void theOpenEventCarriesTheOpenersSpanAndCaptures() {
        Recorder r = new Recorder();
        parse(gemma4(), ALARM, ToolArguments.NONE, r);
        assertEquals(Collections.singletonMap("name", "set_alarm"), r.captures.get("tool_calls"));
        assertEquals(0, r.openSpans.get("tool_calls")[0]);
        assertEquals("<|tool_call>call:set_alarm".length(), r.openSpans.get("tool_calls")[1]);
    }

    @Test
    public void joinConcatenatesTheRepeatsAndRefusesANonStringValue() {
        ResponseTemplate text = ResponseTemplate.parse("{\"start_anchor\": \"<a>\", \"fields\": {\"steps\":"
                + " {\"open\": \"<s>\", \"close\": \"</s>\", \"repeats\": true, \"join\": \" -> \"}}}");
        assertEquals("one -> two", parse(text, "<s>one</s>mid<s>two</s>").get("steps"));
        ResponseTemplate ints = ResponseTemplate.parse("{\"start_anchor\": \"<a>\", \"fields\": {\"content\": {},"
                + " \"steps\": {\"open\": \"<s>\", \"close\": \"</s>\", \"repeats\": true, \"join\": \",\","
                + " \"content\": \"int\"}}}");
        Recorder r = new Recorder();
        Map<String, Object> message = parse(ints, "<s>1</s>", ToolArguments.NONE, r);
        assertTrue(r.events.toString(), r.events.contains("malformed steps: Field 'steps': 'join' requires each match"
                + " to parse to a string, got int."));
        assertFalse(message.containsKey("steps"));
    }

    @Test
    public void aMissingRequiredFieldFailsTheFinish() {
        ResponseTemplate t = ResponseTemplate.parse("{\"start_anchor\": \"<a>\", \"fields\": {\"content\":"
                + " {\"optional\": false}}}");
        ResponseParser p = new ResponseParser(t, "", ToolArguments.NONE, null);
        p.feed("", new Recorder());
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> p.finish(new Recorder()));
        assertTrue(e.getMessage(), e.getMessage().contains("Required response_template fields missing"));
        assertTrue(e.getMessage(), e.getMessage().contains("[content]"));
    }

    @Test
    public void theParserRefusesANullPrefixAndUseAfterFinish() {
        assertThrows(NullPointerException.class, () -> new ResponseParser(gemma4(), null, ToolArguments.NONE, null));
        ResponseParser p = new ResponseParser(gemma4(), "", ToolArguments.NONE, null);
        p.feed("x<turn|>", new Recorder());
        p.finish(new Recorder());
        assertThrows(IllegalStateException.class, () -> p.feed("y", new Recorder()));
        assertThrows(IllegalStateException.class, () -> p.finish(new Recorder()));
    }

    @Test
    @SuppressWarnings("unchecked")
    public void aDelimitedStringArgumentTakesTheCallingToolsSchemaType() {
        ChatTool alarm = ChatTool.builder("set_alarm").parameters((Map<String, Object>) Json.parse("{\"type\":"
                + " \"object\", \"properties\": {\"hour\": {\"type\": \"integer\"}, \"label\": {\"type\": \"string\"}}}"))
                .build();
        Map<String, Object> message = parse(gemma4(),
                "<|tool_call>call:set_alarm{hour:<|\"|>7<|\"|>,label:<|\"|>morning<|\"|>}<tool_call|>",
                ToolArguments.of(Collections.singletonList(alarm)), new Recorder());
        Map<String, Object> args = (Map<String, Object>) function(((List<Object>) message.get("tool_calls")).get(0))
                .get("arguments");
        assertEquals(7L, args.get("hour"));
        assertEquals("morning", args.get("label"));
    }

    @Test
    public void kvLinesXmlInlineAndScalarParsersReadTheirBodies() {
        ResponseTemplate t = ResponseTemplate.parse("{\"start_anchor\": \"<a>\", \"fields\": {"
                + "\"kv\": {\"open\": \"<kv>\", \"close\": \"</kv>\", \"content\": \"kv-lines\"},"
                + "\"tags\": {\"open\": \"<t>\", \"close\": \"</t>\", \"content\": \"xml-inline\", \"content_args\":"
                + " {\"tag_pattern\": \"<p>(?P<key>\\\\w+)=(?P<value>\\\\w+)</p>\"}},"
                + "\"i\": {\"open\": \"<i>\", \"close\": \"</i>\", \"content\": \"int\"},"
                + "\"f\": {\"open\": \"<f>\", \"close\": \"</f>\", \"content\": \"float\"},"
                + "\"b\": {\"open\": \"<b>\", \"close\": \"</b>\", \"content\": \"bool\"}}}");
        Map<String, Object> m = parse(t, "<kv>city: Paris\n\nno separator\ncountry: France</kv>"
                + "<t><p>a=1</p><p>b=2</p></t><i> 42 </i><f>2.5</f><b>TRUE</b>");
        Map<String, Object> kv = new LinkedHashMap<>();
        kv.put("city", "Paris");
        kv.put("country", "France");
        assertEquals(kv, m.get("kv"));
        Map<String, Object> tags = new LinkedHashMap<>();
        tags.put("a", "1");
        tags.put("b", "2");
        assertEquals(tags, m.get("tags"));
        assertEquals(42L, m.get("i"));
        assertEquals(2.5, m.get("f"));
        assertEquals(Boolean.TRUE, m.get("b"));
    }

    // ------------------------------------------------------------------ Python pattern syntax

    @Test
    public void pythonGroupSyntaxKeepsItsMeaning() {
        // underscored names in both named-group spellings, and references by (?P=name), \g<name> and \g<n>
        TemplateRegex.Match m = TemplateRegex.compile("(?P<tool_name>[a-z_]+):(?<arg_1>\\w+)=(?P=arg_1)\\g<tool_name>"
                + "\\g<1>", "test").search("get_weather:x=xget_weatherget_weather", 0, false);
        assertEquals("get_weather", m.captures.get("tool_name"));
        assertEquals("x", m.captures.get("arg_1"));
        assertEquals(Arrays.asList("tool_name", "arg_1"),
                TemplateRegex.compile("(?P<tool_name>a)(?<arg_1>b)", "test").groupNames());
        assertEquals("lookbehinds stay lookbehinds", 1,
                TemplateRegex.compile("(?<=a)b(?<!c)", "test").search("ab", 0, false).start);
        refused("(?P<a>x)(?P<a>y)", "redefined");
        refused("(?P<a-b>x)", "bad character in group name");
        refused("(?P<1a>x)", "bad character in group name");
        refused("(?P=missing)", "unknown group name 'missing'");
    }

    @Test
    public void pythonQuantifiersAnchorsAndClassesKeepTheirMeaning() {
        assertEquals("{,} is {0,}", 4, TemplateRegex.compile("xa{,}", "test").search("xaaa", 0, false).end);
        assertEquals("{,2} is {0,2}", 3, TemplateRegex.compile("xa{,2}", "test").search("xaaa", 0, false).end);
        assertEquals("a brace that starts no quantifier is literal", 0,
                TemplateRegex.compile("\\w{}x{a}{", "test").search("w{}x{a}{", 0, false).start);
        assertNull("\\Z is the very end, not before a final newline",
                TemplateRegex.compile("a\\Z", "test").search("a\n", 0, false));
        assertEquals(0, TemplateRegex.compile("a\\Z", "test").search("a", 0, false).start);
        assertEquals("a class keeps a leading ], [ and && literal", 4,
                TemplateRegex.compile("[]&[]+", "test").search("]&&[", 0, false).end);
        assertEquals("\\w and \\d are Unicode", 5,
                TemplateRegex.compile("\\w+\\d", "test").search("café٣", 0, false).end);
    }

    private static void refused(String pattern, String message) {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> TemplateRegex.compile(pattern, "Field 'x': open_pattern"));
        assertTrue(e.getMessage(), e.getMessage().contains(message));
        assertTrue(e.getMessage(), e.getMessage().startsWith("Field 'x': open_pattern: invalid regex"));
    }
}

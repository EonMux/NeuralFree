package com.eonmux.neuralfree.gguf.chat.tools;

import com.eonmux.neuralfree.gguf.chat.ChatTool;
import com.eonmux.neuralfree.gguf.chat.response.ResponseTemplate;
import com.eonmux.neuralfree.gguf.chat.response.TemplateParsing;
import com.eonmux.neuralfree.utils.json.Json;
import org.junit.Test;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The response-template tool parser against vLLM's {@code hf} parser on the Gemma 4 checkpoint template: the regions
 * separate the calls from the content, a call appears once its region closes, a partial parse's content only grows, a
 * call cut off at the end of the output is parsed as generated, an incomplete or malformed call is dropped, string
 * arguments take the calling tool's schema types, and a template without an implicit field keeps its plain text as
 * content.
 */
public class ResponseTemplateToolParserTest {

    private static final String THINKING = "<|channel>thought\nplan<channel|>";
    private static final String TURN = "<turn|>";
    private static final String ALARM = "<|tool_call>call:set_alarm{hour:7,label:<|\"|>morning<|\"|>}<tool_call|>";
    private static final String WEATHER = "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>";
    private static final String ALARM_CALL = "set_alarm{\"hour\": 7, \"label\": \"morning\"}";

    private static final List<ChatTool> TOOLS = Arrays.asList(
            ChatTool.builder("set_alarm").parameters(schema("{\"type\": \"object\", \"properties\":"
                    + " {\"hour\": {\"type\": \"integer\"}, \"label\": {\"type\": \"string\"}}}")).build(),
            ChatTool.builder("get_weather").parameters(schema("{\"type\": \"object\", \"properties\":"
                    + " {\"city\": {\"type\": \"string\"}}}")).build());

    @SuppressWarnings("unchecked")
    private static Map<String, Object> schema(String json) {
        return (Map<String, Object>) Json.parse(json);
    }

    private static ResponseTemplate gemma4() {
        try (InputStream in = ResponseTemplateToolParserTest.class
                .getResourceAsStream("/response-templates/gemma4.json")) {
            return ResponseTemplate.parse(new String(in.readAllBytes(), StandardCharsets.UTF_8));
        } catch (IOException e) {
            throw new AssertionError(e);
        }
    }

    private static ToolParser parser(ResponseTemplate template, List<ChatTool> tools) {
        return new ResponseTemplateToolParser(TemplateParsing.of("hf", template), tools);
    }

    private static ToolParser parser() {
        return parser(gemma4(), TOOLS);
    }

    /** The parsed calls as {@code name(argumentsJson)}. */
    private static List<String> spelled(ToolParse parse) {
        List<String> out = new ArrayList<>();
        for (ToolCall call : parse.calls()) {
            assertTrue("every read call is complete", call.complete());
            out.add(call.name() + call.arguments());
        }
        return out;
    }

    @Test
    public void theRegionsSeparateTheCallsFromTheContent() {
        assertEquals(Collections.singletonList(ALARM_CALL), spelled(parser().parse(THINKING + ALARM, false)));
        ToolParse parse = parser().parse("Let me check. " + ALARM, false);
        assertEquals("the anchors are not content", "Let me check. ", parse.content());
        assertEquals(1, parse.calls().size());
        assertEquals("Hello", parser().parse("Hello" + TURN, false).content());
        assertEquals("a text that opens no region stays content", "<tool_call>plain text</tool_call>",
                parser().parse("<tool_call>plain text</tool_call>", false).content());
    }

    @Test
    public void aCallAppearsOnlyOnceItsRegionCloses() {
        ToolParser p = parser();
        assertTrue(p.parse("<|tool_call>call:set_alarm{hour:7,", true).calls().isEmpty());
        assertTrue(p.parse("<|tool_call>call:set_alarm{hour:7,label:<|\"|>morning<|\"|>}", true).calls().isEmpty());
        assertTrue("a closer that may still complete waits",
                p.parse("<|tool_call>call:set_alarm{hour:7,label:<|\"|>morning<|\"|>}<tool_ca", true).calls().isEmpty());
        assertEquals(Collections.singletonList(ALARM_CALL), spelled(p.parse(ALARM, true)));
    }

    @Test
    public void aPartialParsesContentOnlyGrows() {
        ToolParser p = parser();
        String text = THINKING + "The answer is 42." + TURN + ALARM;
        String previous = "";
        for (int i = 0; i <= text.length(); i++) {
            String content = p.parse(text.substring(0, i), true).content();
            assertTrue("at " + i + ": " + content, content.startsWith(previous));
            previous = content;
        }
        ToolParse last = p.parse(text, false);
        assertEquals("the reasoning filter is upstream, so the thinking region reads as content",
                "planThe answer is 42.", last.content());
        assertEquals(Collections.singletonList(ALARM_CALL), spelled(last));
    }

    @Test
    public void aCallCutOffAtTheEndIsParsedAsGenerated() {
        // vLLM's finalize closes the open region: a body that parses stands, one that does not is dropped
        ToolParse cut = parser().parse(THINKING + "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}", false);
        assertEquals(Collections.singletonList("get_weather{\"city\": \"Paris\"}"), spelled(cut));
        // vLLM test_gemma4_response_template_drops_incomplete_call
        ToolParse broken = parser().parse("<|tool_call>call:get_weather{city:<|\"|>London", false);
        assertEquals("", broken.content());
        assertTrue(broken.calls().isEmpty());
        assertTrue(parser().parse("<|tool_call>call:set_alarm{oops<tool_call|>", false).calls().isEmpty());
    }

    @Test
    public void contentAfterACallIsKeptAndCallsKeepTheirOrder() {
        // vLLM test_gemma4_response_template_preserves_content_after_tool_call
        ToolParse after = parser().parse(ALARM + "Done." + TURN, false);
        assertEquals("Done.", after.content());
        assertEquals(Collections.singletonList(ALARM_CALL), spelled(after));
        assertEquals(Arrays.asList("get_weather{\"city\": \"Paris\"}", ALARM_CALL),
                spelled(parser().parse(WEATHER + ALARM, false)));
    }

    @Test
    public void aDelimitedStringArgumentTakesTheCallingToolsSchemaType() {
        String call = "<|tool_call>call:set_alarm{hour:<|\"|>7<|\"|>,label:<|\"|>morning<|\"|>}<tool_call|>";
        assertEquals(Collections.singletonList(ALARM_CALL), spelled(parser().parse(call, false)));
        assertEquals("without the tools the string stays a string",
                Collections.singletonList("set_alarm{\"hour\": \"7\", \"label\": \"morning\"}"),
                spelled(parser(gemma4(), Collections.<ChatTool>emptyList()).parse(call, false)));
    }

    @Test
    public void aTemplateWithoutAnImplicitFieldKeepsThePlainTextAsContent() {
        // vLLM test_delimited_template_keeps_plain_content_and_tolerates_required_fields
        ResponseTemplate delimited = ResponseTemplate.parse("{\"start_anchor\": \"<assistant>\", \"fields\": {"
                + "\"thinking\": {\"open\": \"<think>\", \"close\": \"</think>\"},"
                + "\"tool_calls\": {\"open\": [\"<call>\", \"<tool>\"], \"close\": [\"</call>\", \"</tool>\"],"
                + " \"content\": \"json\", \"optional\": false}}}");
        ToolParser p = parser(delimited, TOOLS);
        ToolParse plain = p.parse("<think>plan</think>answer", false);
        assertEquals("plananswer", plain.content());
        assertTrue(plain.calls().isEmpty());
        ToolParse call = p.parse("answer<call>{\"function\": {\"name\": \"get_weather\","
                + " \"arguments\": {\"city\": \"Paris\"}}}</call>", false);
        assertEquals("answer", call.content());
        assertEquals(Collections.singletonList("get_weather{\"city\": \"Paris\"}"), spelled(call));
        assertEquals("the literal openers are the lazy-grammar triggers", Arrays.asList("<call>", "<tool>"),
                p.triggers());
        assertTrue("a regex opener has no literal trigger", parser().triggers().isEmpty());
    }

    @Test
    public void aRegionWithAnIncompleteCallIsDroppedWhole() {
        ResponseTemplate t = ResponseTemplate.parse("{\"start_anchor\": \"<s>\", \"fields\": {"
                + "\"tool_calls\": {\"open\": \"<call>\", \"close\": \"</call>\", \"content\": \"json\"}}}");
        ToolParser p = parser(t, TOOLS);
        assertTrue("no function", p.parse("<call>{\"name\": \"get_weather\"}</call>", false).calls().isEmpty());
        assertTrue("an empty name", p.parse("<call>{\"function\": {\"name\": \"\", \"arguments\": {}}}</call>", false)
                .calls().isEmpty());
        assertTrue("no arguments", p.parse("<call>{\"function\": {\"name\": \"get_weather\"}}</call>", false)
                .calls().isEmpty());
        assertTrue("one incomplete call drops the list", p.parse("<call>[{\"function\": {\"name\": \"get_weather\","
                + " \"arguments\": {}}}, {\"function\": {}}]</call>", false).calls().isEmpty());
        assertEquals("string arguments pass as written",
                Collections.singletonList("get_weather{\"city\":\"Paris\"}"),
                spelled(p.parse("<call>{\"function\": {\"name\": \"get_weather\","
                        + " \"arguments\": \"{\\\"city\\\":\\\"Paris\\\"}\"}}</call>", false)));
    }
}

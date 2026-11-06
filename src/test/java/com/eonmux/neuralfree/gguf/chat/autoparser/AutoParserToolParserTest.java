package com.eonmux.neuralfree.gguf.chat.autoparser;

import com.eonmux.neuralfree.gguf.chat.ChatTool;
import com.eonmux.neuralfree.gguf.chat.tools.ToolCall;
import com.eonmux.neuralfree.gguf.chat.tools.ToolParse;
import com.eonmux.neuralfree.gguf.chat.tools.ToolParser;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** The PEG-driven parser reads the calls of a template with an unknown syntax back out of the generated text. */
public class AutoParserToolParserTest {

    private static ToolParser parser(String template, List<ChatTool> tools) {
        return AutoParserFormat.forTemplate(template).create(tools);
    }

    private static ToolParser parser(String template) {
        return parser(template, AutoParserTestSupport.tools());
    }

    private static final String TOLERANT_TEMPLATE =
            "{# Bailing V3 chat template #}"
            + "{%- for message in messages -%}"
            + "<|{{ message['role'] }}|>{{ message['content'] }}"
            + "{%- if message.get('tool_calls') -%}"
            + "{%- for call in message['tool_calls'] -%}"
            + "<|call|><function={{ call['function']['name'] }}>"
            + "{%- for k, v in call['function']['arguments'].items() -%}"
            + "<arg_key>{{ k }}</arg_key><arg_value>{{ v }}</arg_value>\n"
            + "{%- endfor -%}<|/call|>"
            + "{%- endfor -%}{%- endif -%}<|end|>\n"
            + "{%- endfor -%}"
            + "{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}";

    private static final String OPENAI_WRAPPER_TEMPLATE =
            "{# Respond in the format {\"name\": function name, \"parameters\": ...}. Do not use variables. #}"
            + "{%- for message in messages -%}"
            + "<|{{ message['role'] }}|>{{ message['content'] }}"
            + "{%- if message.get('tool_calls') -%}"
            + "{%- for call in message['tool_calls'] -%}"
            + "{\"name\": \"{{ call['function']['name'] }}\", \"parameters\": "
            + "{{ call['function']['arguments'] | tojson }}}"
            + "{%- endfor -%}{%- endif -%}<|end|>\n"
            + "{%- endfor -%}"
            + "{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}";

    @Test
    public void theOpenAiWrapperFieldIsReadAsTheSameCall() {
        ToolParse parse = parser(OPENAI_WRAPPER_TEMPLATE)
                .parse("{\"type\": \"function\", \"name\": \"get_weather\", \"parameters\": {\"city\": \"SF\"}}",
                        false);
        assertEquals(1, parse.calls().size());
        assertEquals("get_weather", parse.calls().get(0).name());
        assertEquals("{\"city\":\"SF\"}", parse.calls().get(0).arguments());
    }

    @Test
    public void aTolerantFormatReadsWhitespaceBeforeTheCallEnd() {
        ToolParse parse = parser(TOLERANT_TEMPLATE)
                .parse("<|call|><function=get_weather><arg_key>city</arg_key><arg_value>SF</arg_value>\n"
                        + "<|/call|>", false);
        assertEquals(1, parse.calls().size());
        assertEquals("get_weather", parse.calls().get(0).name());
        assertEquals("{\"city\":\"SF\"}", parse.calls().get(0).arguments());
    }

    @Test
    public void tagWithJsonReadsTheNameAndItsObject() {
        ToolParse parse = parser(AutoParserTestSupport.TAG_JSON_TEMPLATE)
                .parse("Hello<|call|><function=get_weather>{\"city\": \"SF\"}</function><|/call|>", false);
        assertEquals("Hello", parse.content());
        assertEquals(1, parse.calls().size());
        assertEquals("get_weather", parse.calls().get(0).name());
        assertEquals("{\"city\": \"SF\"}", parse.calls().get(0).arguments());
        assertTrue(parse.calls().get(0).complete());
    }

    @Test
    public void taggedArgumentsAreRebuiltAsAnObject() {
        ToolParse parse = parser(AutoParserTestSupport.TAGGED_TEMPLATE)
                .parse("<|tool|><|call|><function=get_weather><parameter=city>SF</parameter>"
                        + "</function><|/call|><|/tool|>", false);
        assertEquals("", parse.content());
        assertEquals(1, parse.calls().size());
        assertEquals("get_weather", parse.calls().get(0).name());
        assertEquals("{\"city\":\"SF\"}", parse.calls().get(0).arguments());
    }

    @Test
    public void jsonNativeReadsTheAnalysedFields() {
        ToolParse parse = parser(AutoParserTestSupport.JSON_NATIVE_TEMPLATE)
                .parse("<|tool_calls|>[{\"tool_name\": \"get_weather\", \"parameters\": {\"city\": \"SF\"}}]"
                        + "<|/tool_calls|>", false);
        assertEquals(1, parse.calls().size());
        assertEquals("get_weather", parse.calls().get(0).name());
        assertEquals("{\"city\":\"SF\"}", parse.calls().get(0).arguments());
    }

    @Test
    public void parallelCallsAreReadInOrder() {
        ToolParse parse = parser(AutoParserTestSupport.TAGGED_TEMPLATE)
                .parse("<|tool|>"
                        + "<|call|><function=get_weather><parameter=city>SF</parameter></function><|/call|>"
                        + "<|call|><function=noop></function><|/call|>"
                        + "<|/tool|>", false);
        assertEquals(2, parse.calls().size());
        assertEquals("get_weather", parse.calls().get(0).name());
        assertEquals("noop", parse.calls().get(1).name());
        assertEquals("{}", parse.calls().get(1).arguments());
    }

    @Test
    public void aPartialCallGrowsIntoTheFinalOne() {
        ToolParser parser = parser(AutoParserTestSupport.TAGGED_TEMPLATE);
        String text = "<|tool|><|call|><function=get_weather><parameter=city>SF</parameter>"
                + "</function><|/call|><|/tool|>";
        String previous = "";
        int callsSeen = 0;
        for (int end = 0; end <= text.length(); end++) {
            ToolParse partial = parser.parse(text.substring(0, end), true);
            if (!partial.calls().isEmpty()) {
                ToolCall call = partial.calls().get(0);
                assertEquals("get_weather", call.name());
                String args = call.arguments();
                assertTrue("partial arguments are a prefix: '" + previous + "' -> '" + args + "'",
                        args.startsWith(previous));
                previous = args;
                callsSeen++;
            }
        }
        assertEquals("get_weather", parser.parse(text, false).calls().get(0).name());
        assertTrue(callsSeen > 0);
    }

    @Test
    public void aPartialJsonCallReportsTheNameAndArgumentsSoFar() {
        ToolParser parser = parser(AutoParserTestSupport.JSON_NATIVE_TEMPLATE);
        ToolParse partial = parser.parse("<|tool_calls|>[{\"tool_name\": \"get_weather\", \"parameters\": "
                + "{\"city\": \"S", true);
        assertEquals(1, partial.calls().size());
        assertEquals("get_weather", partial.calls().get(0).name());
        assertFalse(partial.calls().get(0).complete());
        assertTrue(partial.calls().get(0).arguments().startsWith("{\"city\": \"S"));
    }

    @Test
    public void aPartialNameIsNotReportedAsACall() {
        ToolParse parse = parser(AutoParserTestSupport.TAG_JSON_TEMPLATE)
                .parse("<|call|><function=get_", true);
        assertTrue(parse.calls().isEmpty());
    }

    @Test
    public void anUndeclaredToolStaysContent() {
        ToolParse parse = parser(AutoParserTestSupport.TAG_JSON_TEMPLATE)
                .parse("before<|call|><function=delete_everything>{}</function><|/call|>after", false);
        assertTrue(parse.calls().isEmpty());
        assertTrue(parse.content(), parse.content().contains("delete_everything"));
        assertTrue(parse.content(), parse.content().contains("after"));
    }

    @Test
    public void theFormatNeedsAnAnalysedTemplate() {
        ToolParser.Factory row = com.eonmux.neuralfree.gguf.chat.tools.ToolParserRegistry.forName(
                AutoParserFormat.NAME);
        assertThrows(IllegalArgumentException.class, () -> row.create(AutoParserTestSupport.tools()));
        assertThrows(IllegalArgumentException.class, () -> AutoParserFormat.forModelAnalysis(null));
        assertNull(AutoParserFormat.forTemplate("<tool_call>{{ name }}</tool_call>"));
    }
}

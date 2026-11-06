package com.eonmux.neuralfree.gguf.chat.autoparser;

import com.eonmux.neuralfree.gguf.chat.tools.ToolParse;
import com.eonmux.neuralfree.gguf.chat.tools.ToolParser;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Content the template wraps in its own markers: the analysis finds the markers and the parser removes them, so the
 * assistant's answer does not carry the protocol's delimiters.
 */
public class AutoParserContentTest {

    /** Content is wrapped only when it is non-empty, so the with-tools probe shows the markers in the diff. */
    private static final String WRAPPED_CONTENT =
            "{%- for message in messages -%}"
            + "<|{{ message['role'] }}|>"
            + "{%- if message['role'] == 'assistant' and message['content'] -%}"
            + "<|response|>{{ message['content'] }}<|/response|>"
            + "{%- else -%}{{ message['content'] }}{%- endif -%}"
            + "{%- if message.get('tool_calls') -%}"
            + "{%- for call in message['tool_calls'] -%}"
            + "<|call|><function={{ call['function']['name'] }}>"
            + "{{ call['function']['arguments'] | tojson }}</function><|/call|>"
            + "{%- endfor -%}"
            + "{%- endif -%}"
            + "<|end|>\n"
            + "{%- endfor -%}"
            + "{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}";

    @Test
    public void theAnalysisFindsTheContentMarkers() {
        AutoParserAnalysis analysis = AutoParser.analyze(WRAPPED_CONTENT);
        assertEquals(AutoParserAnalysis.ContentMode.ALWAYS_WRAPPED, analysis.content().mode());
        assertEquals("<|response|>", analysis.content().start());
        assertEquals("<|/response|>", analysis.content().end());
    }

    @Test
    public void theParserRemovesTheContentMarkers() {
        ToolParser parser = AutoParserFormat.forTemplate(WRAPPED_CONTENT)
                .create(AutoParserTestSupport.tools());
        ToolParse parse = parser.parse("<|response|>Hello<|/response|>"
                + "<|call|><function=get_weather>{\"city\": \"SF\"}</function><|/call|>", false);
        assertEquals("Hello", parse.content());
        assertEquals(1, parse.calls().size());
        assertEquals("get_weather", parse.calls().get(0).name());
    }

    @Test
    public void contentWithoutCallsLosesTheMarkersToo() {
        ToolParser parser = AutoParserFormat.forTemplate(WRAPPED_CONTENT)
                .create(AutoParserTestSupport.tools());
        ToolParse parse = parser.parse("<|response|>Just an answer<|/response|>", false);
        assertTrue(parse.calls().isEmpty());
        assertEquals("Just an answer", parse.content());
    }

    @Test
    public void aMarkerStillArrivingIsHeldBack() {
        ToolParser parser = AutoParserFormat.forTemplate(WRAPPED_CONTENT)
                .create(AutoParserTestSupport.tools());
        assertEquals("", parser.parse("<|respons", true).content());
        assertEquals("Hello", parser.parse("<|response|>Hello<|/resp", true).content());
        assertEquals("Hello", parser.parse("<|response|>Hello<|/response|>", true).content());
    }
}

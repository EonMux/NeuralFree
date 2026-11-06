package com.eonmux.neuralfree.gguf.chat.autoparser;

import com.eonmux.neuralfree.gguf.chat.tools.ToolParse;
import com.eonmux.neuralfree.gguf.chat.tools.ToolParser;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * A format that carries a call id between the function name and the arguments: the analysis finds the markers and
 * the parser reads the id into the {@link com.eonmux.neuralfree.gguf.chat.tools.ToolCall}.
 */
public class AutoParserCallIdTest {

    private static final String CALL_ID_TEMPLATE =
            "{%- for message in messages -%}"
            + "<|{{ message['role'] }}|>{{ message['content'] }}"
            + "{%- if message.get('tool_calls') -%}"
            + "{%- for call in message['tool_calls'] -%}"
            + "<|call|><function={{ call['function']['name'] }}><|id|>{{ call['id'] }}<|/id|>"
            + "{{ call['function']['arguments'] | tojson }}</function><|/call|>"
            + "{%- endfor -%}"
            + "{%- endif -%}"
            + "<|end|>\n"
            + "{%- endfor -%}"
            + "{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}";

    @Test
    public void theCallIdMarkersAreAnalysed() {
        AutoParserAnalysis analysis = AutoParser.analyze(CALL_ID_TEMPLATE);
        assertEquals(AutoParserAnalysis.ToolFormat.TAG_WITH_JSON, analysis.tools().mode());
        assertEquals(AutoParserAnalysis.CallIdPosition.BETWEEN_FUNC_AND_ARGS,
                analysis.tools().callIdPosition());
        assertEquals("<|id|>", analysis.tools().callIdPrefix());
        assertEquals("<|/id|>", analysis.tools().callIdSuffix());
    }

    @Test
    public void theParserReadsTheCallId() {
        ToolParser parser = AutoParserFormat.forTemplate(CALL_ID_TEMPLATE)
                .create(AutoParserTestSupport.tools());
        ToolParse parse = parser.parse("<|call|><function=get_weather><|id|>call_42<|/id|>"
                + "{\"city\": \"SF\"}</function><|/call|>", false);
        assertEquals(1, parse.calls().size());
        assertEquals("get_weather", parse.calls().get(0).name());
        assertEquals("call_42", parse.calls().get(0).id());
        assertEquals("{\"city\": \"SF\"}", parse.calls().get(0).arguments());
    }

    private static final String PRE_FUNC_ID_TEMPLATE =
            "{%- for message in messages -%}"
            + "<|{{ message['role'] }}|>{{ message['content'] }}"
            + "{%- if message.get('tool_calls') -%}"
            + "{%- for call in message['tool_calls'] -%}"
            + "<|call|><|id|>{{ call['id'] }}<|/id|><function={{ call['function']['name'] }}>"
            + "{{ call['function']['arguments'] | tojson }}</function><|/call|>"
            + "{%- endfor -%}"
            + "{%- endif -%}"
            + "<|end|>\n"
            + "{%- endfor -%}"
            + "{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}";

    private static final String POST_ARGS_ID_TEMPLATE =
            "{%- for message in messages -%}"
            + "<|{{ message['role'] }}|>{{ message['content'] }}"
            + "{%- if message.get('tool_calls') -%}"
            + "{%- for call in message['tool_calls'] -%}"
            + "<|call|><function={{ call['function']['name'] }}>"
            + "{{ call['function']['arguments'] | tojson }}<|id|>{{ call['id'] }}<|/id|>"
            + "</function><|/call|>"
            + "{%- endfor -%}"
            + "{%- endif -%}"
            + "<|end|>\n"
            + "{%- endfor -%}"
            + "{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}";

    @Test
    public void aCallIdBeforeTheFunctionNameIsAnalysed() {
        AutoParserAnalysis analysis = AutoParser.analyze(PRE_FUNC_ID_TEMPLATE);
        assertEquals(AutoParserAnalysis.ToolFormat.TAG_WITH_JSON, analysis.tools().mode());
        assertEquals(AutoParserAnalysis.CallIdPosition.PRE_FUNC_NAME,
                analysis.tools().callIdPosition());
        assertEquals("<|id|>", analysis.tools().callIdPrefix());
        assertEquals("<|/id|>", analysis.tools().callIdSuffix());
        assertEquals("<function=", analysis.tools().functionNamePrefix());
    }

    @Test
    public void theParserReadsAnIdBeforeTheFunctionName() {
        ToolParser parser = AutoParserFormat.forTemplate(PRE_FUNC_ID_TEMPLATE)
                .create(AutoParserTestSupport.tools());
        ToolParse parse = parser.parse("<|call|><|id|>call_42<|/id|><function=get_weather>"
                + "{\"city\": \"SF\"}</function><|/call|>", false);
        assertEquals(1, parse.calls().size());
        assertEquals("get_weather", parse.calls().get(0).name());
        assertEquals("call_42", parse.calls().get(0).id());
    }

    @Test
    public void aCallIdAfterTheArgumentsIsAnalysed() {
        AutoParserAnalysis analysis = AutoParser.analyze(POST_ARGS_ID_TEMPLATE);
        assertEquals(AutoParserAnalysis.ToolFormat.TAG_WITH_JSON, analysis.tools().mode());
        assertEquals(AutoParserAnalysis.CallIdPosition.POST_ARGS, analysis.tools().callIdPosition());
        assertEquals("<|id|>", analysis.tools().callIdPrefix());
        assertEquals("<|/id|>", analysis.tools().callIdSuffix());
    }

    @Test
    public void theParserReadsAnIdAfterTheArguments() {
        ToolParser parser = AutoParserFormat.forTemplate(POST_ARGS_ID_TEMPLATE)
                .create(AutoParserTestSupport.tools());
        ToolParse parse = parser.parse("<|call|><function=get_weather>"
                + "{\"city\": \"SF\"}<|id|>call_42<|/id|></function><|/call|>", false);
        assertEquals(1, parse.calls().size());
        assertEquals("get_weather", parse.calls().get(0).name());
        assertEquals("call_42", parse.calls().get(0).id());
        assertEquals("{\"city\": \"SF\"}", parse.calls().get(0).arguments());
    }
}

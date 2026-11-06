package com.eonmux.neuralfree.gguf.chat.tools;

import com.eonmux.neuralfree.gguf.chat.autoparser.AutoParserFormat;
import com.eonmux.neuralfree.gguf.grammar.CompiledConstraint;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.gguf.grammar.GbnfBuilder;
import org.junit.Test;


import java.util.List;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * The autoparser's grammar comes from the same PEG definition that reads the calls: it admits exactly the declared
 * tools in the template's own syntax, and its lazy form starts at the format's call opener.
 */
public class AutoParserGrammarTest {

    private static final String TAG_WITH_JSON =
            "{%- for message in messages -%}"
            + "<|{{ message['role'] }}|>{{ message['content'] }}"
            + "{%- if message.get('tool_calls') -%}"
            + "{%- for call in message['tool_calls'] -%}"
            + "<|call|><function={{ call['function']['name'] }}>"
            + "{{ call['function']['arguments'] | tojson }}</function><|/call|>"
            + "{%- endfor -%}"
            + "{%- endif -%}<|end|>\n"
            + "{%- endfor -%}"
            + "{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}";

    private static final String TAGGED =
            "{%- for message in messages -%}"
            + "<|{{ message['role'] }}|>{{ message['content'] }}"
            + "{%- if message.get('tool_calls') -%}<|tool|>"
            + "{%- for call in message['tool_calls'] -%}"
            + "<|call|><function={{ call['function']['name'] }}>"
            + "{%- for k, v in call['function']['arguments'].items() -%}"
            + "<parameter={{ k }}>{{ v }}</parameter>"
            + "{%- endfor -%}</function><|/call|>"
            + "{%- endfor -%}<|/tool|>"
            + "{%- endif -%}<|end|>\n"
            + "{%- endfor -%}"
            + "{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}";

    private static final String JSON_NATIVE =
            "{%- for message in messages -%}"
            + "<|{{ message['role'] }}|>{{ message['content'] }}"
            + "{%- if message.get('tool_calls') -%}"
            + "<|tool_calls|>[{% for call in message['tool_calls'] %}"
            + "{\"tool_name\": \"{{ call['function']['name'] }}\", \"parameters\": "
            + "{{ call['function']['arguments'] | tojson }}}"
            + "{% if not loop.last %}, {% endif %}{% endfor %}]<|/tool_calls|>"
            + "{%- endif -%}<|end|>\n"
            + "{%- endfor -%}"
            + "{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}";

    private static final String CALL_ID =
            "{%- for message in messages -%}"
            + "<|{{ message['role'] }}|>{{ message['content'] }}"
            + "{%- if message.get('tool_calls') -%}"
            + "{%- for call in message['tool_calls'] -%}"
            + "<|call|><function={{ call['function']['name'] }}><|id|>{{ call['id'] }}<|/id|>"
            + "{{ call['function']['arguments'] | tojson }}</function><|/call|>"
            + "{%- endfor -%}"
            + "{%- endif -%}<|end|>\n"
            + "{%- endfor -%}"
            + "{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}";

    private static AutoParserFormat format(String template) {
        return (AutoParserFormat) AutoParserFormat.forTemplate(template);
    }

    private static CompiledConstraint forcing(String template, boolean parallel) {
        return GbnfBuilder.compile(format(template).forcing(ToolTestSupport.tools(), parallel));
    }

    @Test
    public void tagWithJsonGrammarAdmitsADeclaredCall() {
        CompiledConstraint c = forcing(TAG_WITH_JSON, false);
        assertTrue(ToolTestSupport.accepts(c,
                "<|call|><function=get_weather>{\"city\": \"SF\"}</function><|/call|>"));
        assertFalse(ToolTestSupport.accepts(c,
                "<|call|><function=delete_everything>{}</function><|/call|>"));
    }

    @Test
    public void taggedGrammarAdmitsADeclaredCall() {
        CompiledConstraint c = forcing(TAGGED, false);
        assertTrue(ToolTestSupport.accepts(c, "<|tool|><|call|><function=get_weather>"
                + "<parameter=city>SF</parameter></function><|/call|><|/tool|>"));
        assertFalse(ToolTestSupport.accepts(c, "<|tool|><|call|><function=delete_everything>"
                + "</function><|/call|><|/tool|>"));
    }

    @Test
    public void taggedGrammarAdmitsParallelCalls() {
        CompiledConstraint parallel = forcing(TAGGED, true);
        assertTrue(ToolTestSupport.accepts(parallel, "<|tool|>"
                + "<|call|><function=get_weather><parameter=city>SF</parameter></function><|/call|>"
                + "<|call|><function=noop></function><|/call|><|/tool|>"));
        CompiledConstraint single = forcing(TAGGED, false);
        assertFalse(ToolTestSupport.accepts(single, "<|tool|>"
                + "<|call|><function=get_weather><parameter=city>SF</parameter></function><|/call|>"
                + "<|call|><function=noop></function><|/call|><|/tool|>"));
    }

    @Test
    public void jsonNativeGrammarAdmitsTheAnalysedFields() {
        CompiledConstraint c = forcing(JSON_NATIVE, false);
        assertTrue(ToolTestSupport.accepts(c,
                "<|tool_calls|>[{\"tool_name\": \"get_weather\", \"parameters\": {\"city\": \"SF\"}}]"
                        + "<|/tool_calls|>"));
        assertFalse(ToolTestSupport.accepts(c,
                "<|tool_calls|>[{\"tool_name\": \"delete_everything\", \"parameters\": {}}]<|/tool_calls|>"));
        assertFalse("the required argument is missing", ToolTestSupport.accepts(c,
                "<|tool_calls|>[{\"tool_name\": \"get_weather\", \"parameters\": {}}]<|/tool_calls|>"));
    }

    @Test
    public void callIdGrammarAdmitsBothTheIdAndItsAbsence() {
        CompiledConstraint c = forcing(CALL_ID, false);
        assertTrue(ToolTestSupport.accepts(c, "<|call|><function=get_weather><|id|>call_42<|/id|>"
                + "{\"city\": \"SF\"}</function><|/call|>"));
        assertTrue(ToolTestSupport.accepts(c,
                "<|call|><function=get_weather>{\"city\": \"SF\"}</function><|/call|>"));
        assertFalse(ToolTestSupport.accepts(c, "<|call|><function=delete_everything><|id|>call_42<|/id|>"
                + "{\"city\": \"SF\"}</function><|/call|>"));
    }

    private static final String PRE_FUNC_ID =
            "{%- for message in messages -%}"
            + "<|{{ message['role'] }}|>{{ message['content'] }}"
            + "{%- if message.get('tool_calls') -%}"
            + "{%- for call in message['tool_calls'] -%}"
            + "<|call|><|id|>{{ call['id'] }}<|/id|><function={{ call['function']['name'] }}>"
            + "{{ call['function']['arguments'] | tojson }}</function><|/call|>"
            + "{%- endfor -%}{%- endif -%}<|end|>\n"
            + "{%- endfor -%}"
            + "{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}";

    private static final String POST_ARGS_ID =
            "{%- for message in messages -%}"
            + "<|{{ message['role'] }}|>{{ message['content'] }}"
            + "{%- if message.get('tool_calls') -%}"
            + "{%- for call in message['tool_calls'] -%}"
            + "<|call|><function={{ call['function']['name'] }}>"
            + "{{ call['function']['arguments'] | tojson }}<|id|>{{ call['id'] }}<|/id|>"
            + "</function><|/call|>"
            + "{%- endfor -%}{%- endif -%}<|end|>\n"
            + "{%- endfor -%}"
            + "{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}";

    @Test
    public void theGrammarAdmitsAnIdBeforeTheFunctionNameAndItsAbsence() {
        CompiledConstraint c = forcing(PRE_FUNC_ID, false);
        assertTrue(ToolTestSupport.accepts(c,
                "<|call|><|id|>call_42<|/id|><function=get_weather>{\"city\": \"SF\"}</function><|/call|>"));
        assertTrue(ToolTestSupport.accepts(c,
                "<|call|><function=get_weather>{\"city\": \"SF\"}</function><|/call|>"));
        assertFalse(ToolTestSupport.accepts(c,
                "<|call|><|id|>call_42<|/id|><function=delete_everything>{}</function><|/call|>"));
    }

    @Test
    public void theGrammarAdmitsAnIdAfterTheArgumentsAndItsAbsence() {
        CompiledConstraint c = forcing(POST_ARGS_ID, false);
        assertTrue(ToolTestSupport.accepts(c,
                "<|call|><function=get_weather>{\"city\": \"SF\"}<|id|>call_42<|/id|></function><|/call|>"));
        assertTrue(ToolTestSupport.accepts(c,
                "<|call|><function=get_weather>{\"city\": \"SF\"}</function><|/call|>"));
    }

    @Test
    public void theLazyGrammarStartsAtTheCallOpener() {
        AutoParserFormat f = format(TAGGED);
        List<String> triggers = f.create(ToolTestSupport.tools()).triggers();
        assertFalse(triggers.isEmpty());
        ConstraintSpec lazy = f.lazy(ToolTestSupport.tools(), triggers);
        assertNotNull(lazy);
        assertNotNull(lazy.triggers());
    }

    private static final String OPENAI_WRAPPER =
            "{# Respond in the format {\"name\": function name, \"parameters\": ...}. Do not use variables. #}"
            + "{%- for message in messages -%}"
            + "<|{{ message['role'] }}|>{{ message['content'] }}"
            + "{%- if message.get('tool_calls') -%}"
            + "{%- for call in message['tool_calls'] -%}"
            + "{\"name\": \"{{ call['function']['name'] }}\", \"parameters\": "
            + "{{ call['function']['arguments'] | tojson }}}"
            + "{% if not loop.last %}\n{% endif %}"
            + "{%- endfor -%}{%- endif -%}<|end|>\n"
            + "{%- endfor -%}"
            + "{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}";

    @Test
    public void theOpenAiWrapperFieldIsOptionalInTheGrammar() {
        CompiledConstraint c = forcing(OPENAI_WRAPPER, false);
        assertTrue(ToolTestSupport.accepts(c,
                "{\"type\": \"function\", \"name\": \"get_weather\", \"parameters\": {\"city\": \"SF\"}}"));
        assertTrue(ToolTestSupport.accepts(c,
                "{\"name\": \"get_weather\", \"parameters\": {\"city\": \"SF\"}}"));
        assertFalse(ToolTestSupport.accepts(c,
                "{\"type\": \"function\", \"name\": \"delete_everything\", \"parameters\": {}}"));
    }

    @Test
    public void theLazyTriggersCarryTheOpenAiWrapper() {
        AutoParserFormat f = (AutoParserFormat) AutoParserFormat.forTemplate(OPENAI_WRAPPER);
        List<String> triggers = f.create(ToolTestSupport.tools()).triggers();
        assertTrue(triggers.toString(), triggers.contains("{\"type\": \"function\","));
    }

    private static final String TOLERANT =
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

    @Test
    public void aTolerantFormatAdmitsWhitespaceBeforeTheCallEnd() {
        CompiledConstraint c = forcing(TOLERANT, false);
        assertTrue(ToolTestSupport.accepts(c, "<|call|><function=get_weather>"
                + "<arg_key>city</arg_key><arg_value>SF</arg_value><|/call|>"));
        assertTrue(ToolTestSupport.accepts(c, "<|call|><function=get_weather>"
                + "<arg_key>city</arg_key><arg_value>SF</arg_value>\n<|/call|>"));
    }

    @Test
    public void aTemplateWithoutToolSyntaxHasNoGrammar() {
        assertNull(AutoParserFormat.forTemplate("<tool_call>{{ name }}</tool_call>"));
    }
}

package com.eonmux.neuralfree.gguf.chat.tools;

import com.eonmux.neuralfree.gguf.chat.autoparser.AutoParserFormat;
import com.eonmux.neuralfree.gguf.grammar.CompiledConstraint;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.gguf.grammar.GbnfBuilder;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * The structured-output grammar an autoparsed template emits: the request's JSON schema bare or in a
 * {@code ```json} fence, the form llama.cpp's autoparser constrains {@code response_format} to.
 */
public class AutoParserResponseFormatTest {

    private static final String UNKNOWN_TEMPLATE =
            "{%- for message in messages -%}"
            + "<|{{ message['role'] }}|>{{ message['content'] }}<|end|>\n"
            + "{%- endfor -%}"
            + "{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}";

    private static final String TOOL_TEMPLATE =
            "{%- for message in messages -%}"
            + "<|{{ message['role'] }}|>{{ message['content'] }}"
            + "{%- if message.get('tool_calls') -%}"
            + "{%- for call in message['tool_calls'] -%}"
            + "<|call|><function={{ call['function']['name'] }}>"
            + "{{ call['function']['arguments'] | tojson }}</function><|/call|>"
            + "{%- endfor -%}{%- endif -%}<|end|>\n"
            + "{%- endfor -%}"
            + "{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}";

    private static final String HERMES_TEMPLATE =
            "{%- for message in messages -%}"
            + "<|{{ message['role'] }}|>{{ message['content'] }}"
            + "{%- if message.get('tool_calls') -%}"
            + "{%- for call in message['tool_calls'] -%}"
            + "<tool_call>{\"name\": \"{{ call['function']['name'] }}\", \"arguments\": "
            + "{{ call['function']['arguments'] | tojson }}}</tool_call>"
            + "{%- endfor -%}{%- endif -%}<|end|>\n"
            + "{%- endfor -%}"
            + "{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}";

    private static final String SCHEMA =
            "{\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}},"
            + " \"required\": [\"city\"], \"additionalProperties\": false}";

    private static ConstraintSpec constraint() {
        AutoParserFormat format = (AutoParserFormat) AutoParserFormat.forTemplate(TOOL_TEMPLATE);
        assertNotNull(format);
        ConstraintSpec spec = format.responseFormat(SCHEMA);
        assertNotNull(spec);
        assertEquals(ConstraintSpec.Kind.GBNF, spec.kind());
        return spec;
    }

    private static boolean accepts(String text) {
        return ToolTestSupport.accepts(GbnfBuilder.compile(constraint()), text);
    }

    @Test
    public void theBareSchemaValueIsAdmitted() {
        assertTrue(accepts("{\"city\": \"SF\"}"));
        assertTrue(accepts(" {\"city\": \"SF\"} "));
    }

    @Test
    public void theFencedSchemaValueIsAdmitted() {
        assertTrue(accepts("```json\n{\"city\": \"SF\"}\n```"));
        assertTrue(accepts("```json {\"city\": \"SF\"} ```"));
    }

    @Test
    public void aValueOffTheSchemaIsRejected() {
        assertFalse(accepts("{\"city\": 42}"));
        assertFalse(accepts("{\"town\": \"SF\"}"));
        assertFalse(accepts("not json"));
        assertFalse(accepts("```json\nnot json\n```"));
    }

    @Test
    public void aKnownFormatIsNotAutoparsed() {
        assertNull(AutoParserFormat.selected("hermes", UNKNOWN_TEMPLATE));
        assertNull(AutoParserFormat.selected("auto", "<tool_call>{{\"name\": x}}</tool_call>"));
        assertNotNull(AutoParserFormat.selected("auto", UNKNOWN_TEMPLATE));
        assertNotNull(AutoParserFormat.selected("autoparser", UNKNOWN_TEMPLATE));
        assertNull("auto keeps the hand-written parser's own constraint path",
                AutoParserFormat.selected("auto", HERMES_TEMPLATE));
        assertNotNull("the explicit name applies even where a hand-written row reads the template",
                AutoParserFormat.selected("autoparser", HERMES_TEMPLATE));
    }

    @Test
    public void aJsonObjectRequestUsesTheGenericObjectValue() {
        assertTrue(accepts("{\"city\": \"SF\"}"));
    }
}

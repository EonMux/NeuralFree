package com.eonmux.neuralfree.gguf.chat.autoparser;

import com.eonmux.neuralfree.gguf.chat.ChatTool;

import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Shared fixtures: three synthetic templates with an unknown syntax and the request's tools. */
final class AutoParserTestSupport {

    /**
     * A tag format whose call is a function tag followed by a JSON argument object:
     * {@code <|call|><function=NAME>{json}</function><|/call|>}, with reasoning in
     * {@code <|think|>...<|/think|>} and no content markers.
     */
    static final String TAG_JSON_TEMPLATE =
            "{%- for message in messages -%}"
            + "<|{{ message['role'] }}|>{{ message['content'] }}"
            + "{%- if message.get('reasoning_content') %}<|think|>{{ message['reasoning_content'] }}"
            + "<|/think|>{% endif -%}"
            + "{%- if message.get('tool_calls') -%}"
            + "{%- for call in message['tool_calls'] -%}"
            + "<|call|><function={{ call['function']['name'] }}>"
            + "{{ call['function']['arguments'] | tojson }}</function><|/call|>"
            + "{%- endfor -%}"
            + "{%- endif -%}"
            + "<|end|>\n"
            + "{%- endfor -%}"
            + "{%- if add_generation_prompt -%}<|assistant|>"
            + "{% if enable_thinking %}<|think|>{% endif %}{%- endif -%}";

    /**
     * A tag format whose arguments are tagged:
     * {@code <|tool|><|call|><function=NAME><parameter=K>V</parameter></function><|/call|><|/tool|>}.
     */
    static final String TAGGED_TEMPLATE =
            "{%- for message in messages -%}"
            + "<|{{ message['role'] }}|>{{ message['content'] }}"
            + "{%- if message.get('reasoning_content') %}<|think|>{{ message['reasoning_content'] }}"
            + "<|/think|>{% endif -%}"
            + "{%- if message.get('tool_calls') -%}"
            + "<|tool|>"
            + "{%- for call in message['tool_calls'] -%}"
            + "<|call|><function={{ call['function']['name'] }}>"
            + "{%- for k, v in call['function']['arguments'].items() -%}"
            + "<parameter={{ k }}>{{ v }}</parameter>"
            + "{%- endfor -%}"
            + "</function><|/call|>"
            + "{%- endfor -%}"
            + "<|/tool|>"
            + "{%- endif -%}"
            + "<|end|>\n"
            + "{%- endfor -%}"
            + "{%- if add_generation_prompt -%}<|assistant|>"
            + "{% if enable_thinking %}<|think|>{% endif %}{%- endif -%}";

    /** A JSON-native format: {@code <|tool_calls|>[{"tool_name": ..., "parameters": {...}}]<|/tool_calls|>}. */
    static final String JSON_NATIVE_TEMPLATE =
            "{%- for message in messages -%}"
            + "<|{{ message['role'] }}|>{{ message['content'] }}"
            + "{%- if message.get('reasoning_content') %}<|think|>{{ message['reasoning_content'] }}"
            + "<|/think|>{% endif -%}"
            + "{%- if message.get('tool_calls') -%}"
            + "<|tool_calls|>[{% for call in message['tool_calls'] %}"
            + "{\"tool_name\": \"{{ call['function']['name'] }}\", \"parameters\": "
            + "{{ call['function']['arguments'] | tojson }}}"
            + "{% if not loop.last %}, {% endif %}{% endfor %}]<|/tool_calls|>"
            + "{%- endif -%}"
            + "<|end|>\n"
            + "{%- endfor -%}"
            + "{%- if add_generation_prompt -%}<|assistant|>"
            + "{% if enable_thinking %}<|think|>{% endif %}{%- endif -%}";

    private AutoParserTestSupport() {
    }

    /** get_weather(city: string, required) and a parameterless noop. */
    static List<ChatTool> tools() {
        Map<String, Object> city = new LinkedHashMap<>();
        city.put("type", "string");
        Map<String, Object> properties = new LinkedHashMap<>();
        properties.put("city", city);
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("type", "object");
        params.put("properties", properties);
        params.put("required", Collections.singletonList("city"));
        params.put("additionalProperties", Boolean.FALSE);
        return Arrays.asList(ChatTool.builder("get_weather").parameters(params).build(),
                ChatTool.builder("noop").build());
    }
}

package com.eonmux.neuralfree.gguf.chat;

import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Properties;

/**
 * Configured chat template variables: llama.cpp's {@code --chat-template-kwargs} and vLLM's
 * {@code --default-chat-template-kwargs}, a JSON object whose entries are bound into every chat
 * prompt's template context. A server applies them as defaults under each request's own variables
 * ({@link ChatRequest#withDefaultTemplateKwargs}); the local console binds them as its
 * {@code chat_template_kwargs}.
 */
public final class ChatTemplateKwargs {

    /** System property holding the variables as one JSON object, such as {@code {"enable_thinking": false}}. */
    public static final String PROPERTY = "neuralfree.chat.template.kwargs";

    private ChatTemplateKwargs() {}

    /**
     * The variables of {@code json}, a JSON object of JSON values, in member order.
     *
     * @param json   the JSON text; not null
     * @param source what carried the text (a flag or property name), named in the error
     * @return an unmodifiable map of deep-copied values
     * @throws IllegalArgumentException when the text is not JSON, is not an object, or holds an empty key
     */
    public static Map<String, Object> parse(String json, String source) {
        if (json == null) throw new IllegalArgumentException(source + " is null");
        Object parsed;
        try {
            parsed = Json.parse(json, Json.Limits.UNTRUSTED);
        } catch (JsonException e) {
            throw new IllegalArgumentException(source + " is not valid JSON: " + e.getMessage(), e);
        }
        if (!(parsed instanceof Map)) {
            throw new IllegalArgumentException(source + " must be a JSON object of template variables, got "
                    + json.trim());
        }
        Map<String, Object> out = new LinkedHashMap<>();
        for (Map.Entry<?, ?> e : ((Map<?, ?>) parsed).entrySet()) {
            String name = String.valueOf(e.getKey());
            if (name.isEmpty()) throw new IllegalArgumentException(source + " holds an empty variable name");
            out.put(name, ChatMessages.copyValue(e.getValue(), source + " key '" + name + "'"));
        }
        return Collections.unmodifiableMap(out);
    }

    /**
     * The variables {@link #PROPERTY} holds in {@code properties}; empty when it is unset or blank.
     *
     * @throws IllegalArgumentException when the property is set but is not a JSON object
     */
    public static Map<String, Object> fromProperties(Properties properties) {
        String raw = properties.getProperty(PROPERTY);
        if (raw == null || raw.trim().isEmpty()) return Collections.emptyMap();
        return parse(raw, PROPERTY);
    }

    /** The variables as one compact JSON object, the text {@link #PROPERTY} holds. */
    public static String toJson(Map<String, ?> variables) {
        return Json.toString(variables, Json.Style.API);
    }
}

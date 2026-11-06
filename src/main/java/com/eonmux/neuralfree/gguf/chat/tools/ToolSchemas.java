package com.eonmux.neuralfree.gguf.chat.tools;

import com.eonmux.neuralfree.gguf.chat.ChatTool;
import com.eonmux.neuralfree.utils.json.Json;

import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The parameter types of a request's tools, for formats that write each argument as text (Qwen3-Coder's
 * {@code <parameter=k>v</parameter>}, GLM's {@code <arg_value>}, MiniMax's {@code <parameter name="k">}): a value
 * whose schema admits only strings stays the text, any other declared type reads the text as JSON and keeps the
 * text when it is no JSON value, as llama.cpp's {@code tool_arg_json_value | tool_arg_string_value} choice does
 * (common/parsers/qwen3-coder.cpp:113-140). A parameter the schema does not declare stays text.
 *
 * <p>{@link #arguments} renders the JSON object text the OpenAI {@code arguments} field carries. Immutable.
 */
public final class ToolSchemas {

    private static final Set<String> STRING_ONLY = Collections.singleton("string");

    /** function name to parameter name to declared JSON types */
    private final Map<String, Map<String, Set<String>>> types;

    private ToolSchemas(Map<String, Map<String, Set<String>>> types) {
        this.types = types;
    }

    /** The schemas of {@code tools}; a tool without a parameter schema types nothing. */
    public static ToolSchemas of(List<ChatTool> tools) {
        Map<String, Map<String, Set<String>>> out = new HashMap<>();
        for (ChatTool tool : tools) {
            Map<String, Set<String>> params = new HashMap<>();
            Map<String, Object> schema = tool.parameters();
            Object props = schema == null ? null : schema.get("properties");
            if (props instanceof Map) {
                for (Map.Entry<?, ?> e : ((Map<?, ?>) props).entrySet()) {
                    params.put(String.valueOf(e.getKey()), typesOf(e.getValue()));
                }
            }
            out.put(tool.name(), params);
        }
        return new ToolSchemas(out);
    }

    /** No tools: every value stays text. */
    public static ToolSchemas none() {
        return new ToolSchemas(Collections.<String, Map<String, Set<String>>>emptyMap());
    }

    /** The JSON value of argument {@code param} of {@code function} written as {@code text}. */
    public Object value(String function, String param, String text) {
        Map<String, Set<String>> params = types.get(function);
        Set<String> t = params == null ? null : params.get(param);
        if (t == null || t.isEmpty() || STRING_ONLY.equals(t)) {
            return text;
        }
        Object parsed = JsonScan.parseOrNull(text.trim());
        if (parsed == null && t.contains("null") && text.trim().equals("null")) {
            return null;
        }
        return parsed == null ? text : parsed;
    }

    /**
     * Whether argument {@code param} of {@code function} may be a string (no declared type, or {@code string} among
     * them): a tagged-text format writes such a value raw, any other as JSON (llama.cpp {@code may_be_string}).
     */
    public boolean textValued(String function, String param) {
        Map<String, Set<String>> params = types.get(function);
        Set<String> t = params == null ? null : params.get(param);
        return t == null || t.isEmpty() || t.contains("string");
    }

    /** The JSON object text of {@code args} (insertion order), the OpenAI {@code arguments} string. */
    public static String arguments(Map<String, Object> args) {
        StringBuilder out = new StringBuilder();
        try {
            Json.write(new LinkedHashMap<>(args), Json.Style.API, out);
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
        return out.toString();
    }

    /** One argument as JSON text: {@code "key":value} without braces, the unit argument streams grow by. */
    static String member(String key, Object value) {
        Map<String, Object> one = new LinkedHashMap<>();
        one.put(key, value);
        String obj = arguments(one);
        return obj.substring(1, obj.length() - 1);
    }

    /** The declared types of one property schema: {@code type} (a name or a list) and the {@code anyOf}/{@code oneOf} branches. */
    private static Set<String> typesOf(Object schema) {
        Set<String> out = new LinkedHashSet<>();
        if (!(schema instanceof Map)) {
            return out;
        }
        Map<?, ?> m = (Map<?, ?>) schema;
        Object type = m.get("type");
        if (type instanceof String) {
            out.add((String) type);
        } else if (type instanceof List) {
            for (Object o : (List<?>) type) {
                out.add(String.valueOf(o));
            }
        }
        for (String key : new String[]{"anyOf", "oneOf"}) {
            if (m.get(key) instanceof List) {
                for (Object branch : (List<?>) m.get(key)) {
                    out.addAll(typesOf(branch));
                }
            }
        }
        if (out.isEmpty() && (m.containsKey("enum") || m.containsKey("const"))) {
            Object sample = m.containsKey("const") ? m.get("const")
                    : (m.get("enum") instanceof List && !((List<?>) m.get("enum")).isEmpty()
                    ? ((List<?>) m.get("enum")).get(0) : null);
            out.add(sample instanceof String ? "string" : "number");
        }
        if (out.isEmpty() && m.containsKey("properties")) {
            out.add("object");
        }
        if (out.isEmpty() && m.containsKey("items")) {
            out.add("array");
        }
        return out;
    }
}

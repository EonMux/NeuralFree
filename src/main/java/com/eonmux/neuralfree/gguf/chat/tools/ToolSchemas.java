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
 * <p>{@link #arguments} renders the JSON object text the OpenAI {@code arguments} field carries. The declared
 * property schemas are kept too, for a format whose values nest by their schema (MiniMax-M3's recursive
 * elements). Immutable.
 */
public final class ToolSchemas {

    /** function name to parameter name to declared JSON types */
    private final Map<String, Map<String, Set<String>>> types;

    /** function name to parameter name to its property schema, for formats whose values nest by schema */
    private final Map<String, Map<String, Object>> schemas;

    private ToolSchemas(Map<String, Map<String, Set<String>>> types,
                        Map<String, Map<String, Object>> schemas) {
        this.types = types;
        this.schemas = schemas;
    }

    /** The schemas of {@code tools}; a tool without a parameter schema types nothing. */
    public static ToolSchemas of(List<ChatTool> tools) {
        Map<String, Map<String, Set<String>>> out = new HashMap<>();
        Map<String, Map<String, Object>> properties = new HashMap<>();
        for (ChatTool tool : tools) {
            Map<String, Set<String>> params = new HashMap<>();
            Map<String, Object> declared = new HashMap<>();
            Map<String, Object> schema = tool.parameters();
            Object props = schema == null ? null : schema.get("properties");
            if (props instanceof Map) {
                for (Map.Entry<?, ?> e : ((Map<?, ?>) props).entrySet()) {
                    params.put(String.valueOf(e.getKey()), typesOf(e.getValue()));
                    declared.put(String.valueOf(e.getKey()), e.getValue());
                }
            }
            out.put(tool.name(), params);
            properties.put(tool.name(), declared);
        }
        return new ToolSchemas(out, properties);
    }

    /** No tools: every value stays text. */
    public static ToolSchemas none() {
        return new ToolSchemas(Collections.<String, Map<String, Set<String>>>emptyMap(),
                Collections.<String, Map<String, Object>>emptyMap());
    }

    /** The declared property schema of argument {@code param} of {@code function}, or null. */
    public Object schemaOf(String function, String param) {
        Map<String, Object> params = schemas.get(function);
        return params == null ? null : params.get(param);
    }

    /** The JSON value of argument {@code param} of {@code function} written as {@code text}. */
    public Object value(String function, String param, String text) {
        return valueOfSchema(schemaOf(function, param), text);
    }

    /**
     * The JSON value of {@code text} under property {@code schema}: a value whose schema admits only strings stays
     * the text, any other declared type reads the text as JSON and keeps the text when it is no JSON value.
     */
    static Object valueOfSchema(Object schema, String text) {
        Set<String> t = schema == null ? Collections.<String>emptySet() : typesOf(schema);
        if (t.isEmpty() || (t.size() == 1 && t.contains("string"))) {
            return text;
        }
        Object parsed = JsonScan.parseOrNull(text.trim());
        if (parsed == null && t.contains("null") && text.trim().equals("null")) {
            return null;
        }
        return parsed == null ? text : parsed;
    }

    /**
     * Whether argument {@code param} of {@code function} may be a string (no declared type, or {@code string}
     * among them): a tagged-text format writes such a value raw, any other as JSON (llama.cpp
     * {@code may_be_string}).
     */
    public boolean textValued(String function, String param) {
        Map<String, Set<String>> params = types.get(function);
        Set<String> t = params == null ? null : params.get(param);
        return t == null || t.isEmpty() || t.contains("string");
    }

    /**
     * Whether one property schema admits a string (or declares nothing): a tagged-text format writes such a
     * value raw, any other as JSON.
     */
    public static boolean textValuedSchema(Object schema) {
        Set<String> t = typesOf(schema);
        return t.isEmpty() || t.contains("string");
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

    /**
     * Whether one property schema declares the object kind with named properties: a format that writes values as
     * nested elements expands such a value instead of JSON text.
     */
    static boolean objectSchema(Object schema) {
        if (!(schema instanceof Map)) {
            return false;
        }
        Map<?, ?> m = (Map<?, ?>) schema;
        Object props = m.get("properties");
        return "object".equals(m.get("type")) && props instanceof Map && !((Map<?, ?>) props).isEmpty();
    }

    /** Whether one property schema declares the array kind. */
    static boolean arraySchema(Object schema) {
        return schema instanceof Map && "array".equals(((Map<?, ?>) schema).get("type"));
    }

    /** The item schema of an array property, or null when it declares none. */
    static Object itemsSchema(Object schema) {
        return schema instanceof Map ? ((Map<?, ?>) schema).get("items") : null;
    }

    /** The schema of property {@code key} of an object schema, or null. */
    static Object propertySchema(Object schema, String key) {
        Object props = schema instanceof Map ? ((Map<?, ?>) schema).get("properties") : null;
        if (!(props instanceof Map)) {
            return null;
        }
        return ((Map<?, ?>) props).get(key);
    }

    /** The required property names of an object schema; empty when it declares none. */
    static List<?> requiredProperties(Object schema) {
        Object required = schema instanceof Map ? ((Map<?, ?>) schema).get("required") : null;
        return required instanceof List ? (List<?>) required : Collections.emptyList();
    }

    /** The declared types of one property schema: {@code type} (a name or a list) and the {@code anyOf}/{@code oneOf} branches. */
    static Set<String> typesOf(Object schema) {
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

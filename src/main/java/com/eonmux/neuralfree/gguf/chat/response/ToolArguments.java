package com.eonmux.neuralfree.gguf.chat.response;

import com.eonmux.neuralfree.gguf.chat.ChatTool;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * The request's tool schemas as a parsed region's call arguments are typed by (vLLM {@code ResponseParser}'s
 * {@code _coerce_tool_calls}): in a value shaped {@code {"function": {"name", "arguments": {...}}}} (or a list of
 * them), each argument written as text takes the first type of its parameter schema it reads as — {@code integer},
 * {@code number}, {@code boolean}, {@code null}, {@code object} or {@code array}, unions and {@code nullable}
 * included — and keeps its text otherwise. Immutable.
 */
public final class ToolArguments {

    /** No schemas: values pass unchanged. */
    public static final ToolArguments NONE = new ToolArguments(Collections.<String, Map<?, ?>>emptyMap());

    private static final Object NO_MATCH = new Object();

    private final Map<String, Map<?, ?>> properties;

    private ToolArguments(Map<String, Map<?, ?>> properties) {
        this.properties = properties;
    }

    /** The parameter schemas of {@code tools}. */
    public static ToolArguments of(List<ChatTool> tools) {
        if (tools == null || tools.isEmpty()) return NONE;
        Map<String, Map<?, ?>> out = new LinkedHashMap<>();
        for (ChatTool t : tools) {
            Object props = t.parameters() == null ? null : t.parameters().get("properties");
            out.put(t.name(), props instanceof Map ? (Map<?, ?>) props : Collections.emptyMap());
        }
        return new ToolArguments(Collections.unmodifiableMap(out));
    }

    /** {@code value} with its call arguments typed; {@code value} itself is updated in place. */
    @SuppressWarnings("unchecked")
    Object coerce(Object value) {
        if (properties.isEmpty()) return value;
        if (value instanceof List) {
            List<Object> list = (List<Object>) value;
            for (int i = 0; i < list.size(); i++) list.set(i, coerce(list.get(i)));
            return list;
        }
        Object fn = value instanceof Map ? ((Map<?, ?>) value).get("function") : null;
        if (!(fn instanceof Map)) return value;
        Object name = ((Map<?, ?>) fn).get("name");
        Object arguments = ((Map<?, ?>) fn).get("arguments");
        if (!(name instanceof String) || !(arguments instanceof Map)) return value;
        Map<?, ?> props = properties.get(name);
        if (props == null || props.isEmpty()) return value;
        for (Map.Entry<String, Object> e : ((Map<String, Object>) arguments).entrySet()) {
            List<String> types = props.containsKey(e.getKey()) ? schemaTypes(props.get(e.getKey())) : null;
            if (types == null || types.isEmpty()) continue;
            Object argument = e.getValue();
            if (argument instanceof String) {
                e.setValue(coerce((String) argument, types));
            } else if (argument instanceof List && !types.contains("array")) {
                List<Object> items = new ArrayList<>();
                for (Object item : (List<?>) argument) {
                    items.add(item instanceof String ? coerce((String) item, types) : item);
                }
                e.setValue(items);
            }
        }
        return value;
    }

    /** The JSON types a schema admits: its {@code type}, its union branches, and {@code null} when nullable. */
    static List<String> schemaTypes(Object schema) {
        List<String> types = new ArrayList<>();
        if (!(schema instanceof Map)) return types;
        Map<?, ?> s = (Map<?, ?>) schema;
        Object declared = s.get("type");
        if (declared instanceof String) types.add((String) declared);
        if (declared instanceof List) {
            for (Object t : (List<?>) declared) {
                if (t instanceof String) types.add((String) t);
            }
        }
        for (String union : new String[]{"anyOf", "oneOf"}) {
            if (s.get(union) instanceof List) {
                for (Object choice : (List<?>) s.get(union)) types.addAll(schemaTypes(choice));
            }
        }
        if (ContentParser.truthy(s.get("nullable")) && !types.contains("null")) types.add("null");
        return types;
    }

    /** {@code raw} as the first type it reads as, else {@code raw} itself. */
    static Object coerce(String raw, List<String> types) {
        for (String type : types) {
            Object v = as(raw, type);
            if (v != NO_MATCH) return v;
        }
        return raw;
    }

    private static Object as(String raw, String type) {
        try {
            switch (type) {
                case "integer": return integer(raw);
                case "number": {
                    double d = ContentParser.decimal(raw);
                    if (Double.isNaN(d) || Double.isInfinite(d)) return NO_MATCH;
                    return d == Math.rint(d) && !raw.contains(".") ? ContentParser.integer(wholeText(d)) : (Object) d;
                }
                case "boolean": {
                    String t = raw.strip().toLowerCase(Locale.ROOT);
                    if (t.equals("true") || t.equals("1")) return Boolean.TRUE;
                    return t.equals("false") || t.equals("0") ? Boolean.FALSE : NO_MATCH;
                }
                case "null": {
                    String t = raw.strip();
                    return t.equals("null") || t.equals("None") ? null : NO_MATCH;
                }
                case "object":
                case "array": {
                    Object decoded = Json.parse(raw, Json.Limits.PYTHON);
                    return (type.equals("object") ? decoded instanceof Map : decoded instanceof List) ? decoded : NO_MATCH;
                }
                default: return NO_MATCH;
            }
        } catch (IllegalArgumentException | JsonException e) {
            return NO_MATCH;
        }
    }

    /** Python's integer reading: {@code int(raw)}, else a whole {@code float(raw)}. */
    private static Object integer(String raw) {
        try {
            return ContentParser.integer(raw);
        } catch (IllegalArgumentException e) {
            double d = ContentParser.decimal(raw);
            if (Double.isNaN(d) || Double.isInfinite(d) || d != Math.rint(d)) return NO_MATCH;
            return ContentParser.integer(wholeText(d));
        }
    }

    private static String wholeText(double d) {
        return new java.math.BigDecimal(d).toBigInteger().toString();
    }
}

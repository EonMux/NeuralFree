package com.eonmux.neuralfree.gguf.chat.response;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A field's {@code transform} template (vLLM {@code _apply_transform}): a JSON value whose strings that are wholly one
 * {@code {name}} or dotted {@code {name.key}} placeholder take the named value of the scope — the parsed
 * {@code content} and the opener's named captures — and whose other strings stay literal. A string mixing a
 * placeholder with text is refused when the template loads. Immutable.
 */
final class Transform {

    private static final Pattern PLACEHOLDER = Pattern.compile("\\{(\\w+(?:\\.\\w+)*)\\}", Pattern.UNICODE_CHARACTER_CLASS);

    private final Object template;

    private Transform(Object template) {
        this.template = template;
    }

    /**
     * The transform {@code template}.
     *
     * @throws IllegalArgumentException for a string that mixes a placeholder with literal text
     */
    static Transform of(String scope, Object template) {
        validate(scope, template);
        return new Transform(template);
    }

    private static void validate(String scope, Object t) {
        if (t instanceof Map) {
            for (Object v : ((Map<?, ?>) t).values()) validate(scope, v);
        } else if (t instanceof List) {
            for (Object v : (List<?>) t) validate(scope, v);
        } else if (t instanceof String && PLACEHOLDER.matcher((String) t).find()
                && !PLACEHOLDER.matcher((String) t).matches()) {
            throw new IllegalArgumentException(scope + ": transform string '" + t + "' mixes a {placeholder} with"
                    + " literal text. Use either a whole-string placeholder (e.g. \"{content}\") or a plain literal;"
                    + " string interpolation is not supported.");
        }
    }

    /** The template as written. */
    Object template() {
        return template;
    }

    /**
     * The template with every placeholder replaced from {@code scope}.
     *
     * @throws IllegalArgumentException for a placeholder the scope lacks, or a dotted key that indexes into a
     *                                  non-object or a missing member
     */
    Object apply(Map<String, Object> scope) {
        return apply(template, scope);
    }

    private static Object apply(Object t, Map<String, Object> scope) {
        if (t instanceof Map) {
            Map<String, Object> out = new LinkedHashMap<>();
            for (Map.Entry<?, ?> e : ((Map<?, ?>) t).entrySet()) {
                out.put(String.valueOf(e.getKey()), apply(e.getValue(), scope));
            }
            return out;
        }
        if (t instanceof List) {
            List<Object> out = new ArrayList<>();
            for (Object v : (List<?>) t) out.add(apply(v, scope));
            return out;
        }
        if (!(t instanceof String)) return t;
        Matcher whole = PLACEHOLDER.matcher((String) t);
        if (!whole.matches()) return t;
        String path = whole.group(1);
        String[] keys = path.split("\\.");
        if (!scope.containsKey(keys[0])) {
            throw new IllegalArgumentException("transform placeholder '{" + path + "}' is not defined. Available: "
                    + new TreeSet<>(scope.keySet()));
        }
        Object value = scope.get(keys[0]);
        for (int i = 1; i < keys.length; i++) {
            if (!(value instanceof Map)) {
                throw new IllegalArgumentException("transform placeholder '{" + path + "}' cannot index into "
                        + typeName(value) + " at '" + keys[i] + "'");
            }
            Map<?, ?> m = (Map<?, ?>) value;
            if (!m.containsKey(keys[i])) {
                throw new IllegalArgumentException("transform placeholder '{" + path + "}' is missing key '" + keys[i]
                        + "'. Available: " + new TreeSet<>(stringKeys(m)));
            }
            value = m.get(keys[i]);
        }
        return value;
    }

    private static List<String> stringKeys(Map<?, ?> m) {
        List<String> out = new ArrayList<>();
        for (Object k : m.keySet()) out.add(String.valueOf(k));
        return out;
    }

    /** The Python type name of a JSON value, for errors. */
    static String typeName(Object v) {
        if (v == null) return "NoneType";
        if (v instanceof String) return "str";
        if (v instanceof Boolean) return "bool";
        if (v instanceof Double) return "float";
        if (v instanceof Number) return "int";
        if (v instanceof List) return "list";
        if (v instanceof Map) return "dict";
        return v.getClass().getSimpleName();
    }
}

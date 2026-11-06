package com.eonmux.neuralfree.gguf.jinja;

import java.util.Collections;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Value-usage statistics of one render, the engine's form of llama.cpp's {@code value_t::stats}
 * (common/jinja/value.h, runtime.cpp with {@code is_get_stats}): which context values the template
 * touched and with which operations. A chat-template capability probe renders with an instance
 * and reads back whether, say, {@code messages[0].content} was read, iterated as a list or tested
 * with {@code is string}.
 *
 * <p>A value counts as used when a variable lookup yields it, when it is the object, key or result
 * of an attribute or item access, when a filter or test is applied to it, when a loop iterates it,
 * and, recursively with its items, when {@code tojson} or {@code string} serialises it. The recorded
 * operations are {@link #ARRAY_ACCESS} (a loop over it or an integer subscript), {@link #OBJECT_ACCESS}
 * (a key or attribute lookup on it), the filter name of an applied filter, the method name of an
 * accessed method, and {@link #TEST_PREFIX} plus the test name of an applied test.</p>
 *
 * <p>Values are tracked by identity, so a probe that asks about one position of its input gives every
 * string there a distinct instance. Rendering converts the context first ({@link JinjaValue#fromJava});
 * {@link #variable} hands out the converted values the template saw, which are the ones to query.</p>
 *
 * <p>Not thread-safe: one instance belongs to one render.</p>
 */
public final class JinjaUsage {

    /** The operation a loop over a value or an integer subscript of it records. */
    public static final String ARRAY_ACCESS = "array_access";
    /** The operation a key or attribute lookup on a value records. */
    public static final String OBJECT_ACCESS = "object_access";
    /** The prefix of the operation an applied test records, as in {@code test_is_string}. */
    public static final String TEST_PREFIX = "test_is_";

    private static final Set<String> DEEP_FILTERS = Collections.unmodifiableSet(
            new HashSet<>(java.util.Arrays.asList("tojson", "string")));

    /** Used values mapped to the operations recorded on them. */
    private final Map<Object, Set<String>> usage = new IdentityHashMap<>();
    private final Map<String, Object> variables = new LinkedHashMap<>();

    /** Whether the render used {@code value}; never for Undefined, which stands for an absent input. */
    public boolean used(Object value) {
        return value != JinjaValue.UNDEFINED && usage.containsKey(value);
    }

    /** Whether the render applied operation {@code op} to {@code value}; never for Undefined. */
    public boolean hasOp(Object value, String op) {
        if (value == JinjaValue.UNDEFINED) return false;
        Set<String> ops = usage.get(value);
        return ops != null && ops.contains(op);
    }

    /** The converted value of context variable {@code name} as the template saw it, or Undefined. */
    public Object variable(String name) {
        return variables.containsKey(name) ? variables.get(name) : JinjaValue.UNDEFINED;
    }

    void bind(String name, Object converted) {
        variables.put(name, converted);
    }

    void markUsed(Object value) {
        if (!usage.containsKey(value)) usage.put(value, new HashSet<String>());
    }

    void op(Object value, String op) {
        Set<String> ops = usage.get(value);
        if (ops == null) {
            ops = new HashSet<>();
            usage.put(value, ops);
        }
        ops.add(op);
    }

    /** A filter applied to {@code input}; {@code tojson} and {@code string} use its whole tree. */
    void filter(String name, Object input) {
        op(input, name);
        if (DEEP_FILTERS.contains(name)) markDeep(input, Collections.newSetFromMap(new IdentityHashMap<>()));
    }

    /** A test applied to {@code input}. */
    void test(String name, Object input) {
        op(input, TEST_PREFIX + name);
    }

    /** A loop over {@code iterable}: an array access, and an object access when it is a dict. */
    void iterate(Object iterable) {
        op(iterable, ARRAY_ACCESS);
        if (JinjaValue.isDict(iterable)) op(iterable, OBJECT_ACCESS);
    }

    /**
     * An attribute or item access {@code object[key] -> value}, recorded as llama.cpp's member access
     * does: the three values are used; a string key naming a method of a string or list records that
     * method; the object records an object access for a dict object or a string, float, list or None
     * key, otherwise an array access for an integer or boolean key.
     */
    void access(Object object, Object key, Object value) {
        markUsed(value);
        markUsed(object);
        markUsed(key);
        if (key instanceof String && (object instanceof String || object instanceof List)) {
            op(object, (String) key);
        }
        if (JinjaValue.isDict(object) || key instanceof String || key instanceof Double || key instanceof List
                || key == null) {
            op(object, OBJECT_ACCESS);
        } else if (key instanceof Long || key instanceof Boolean) {
            op(object, ARRAY_ACCESS);
        }
    }

    private void markDeep(Object value, Set<Object> visited) {
        markUsed(value);
        if (!(value instanceof Map || value instanceof List) || !visited.add(value)) return;
        if (value instanceof Map) {
            for (Map.Entry<?, ?> e : ((Map<?, ?>) value).entrySet()) {
                markDeep(e.getKey(), visited);
                markDeep(e.getValue(), visited);
            }
        } else {
            for (Object item : (List<?>) value) markDeep(item, visited);
        }
    }
}

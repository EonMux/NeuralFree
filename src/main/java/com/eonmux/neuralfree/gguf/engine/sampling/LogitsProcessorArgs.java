package com.eonmux.neuralfree.gguf.engine.sampling;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * The per-request arguments of a custom logits processor: the members of the request's {@code vllm_xargs} object
 * (vLLM's {@code SamplingParams.extra_args}), as the processor factories read them.
 *
 * <p>A value is a string, a boolean, an integer (held as a {@code Long}), a number (held as a {@code Double}) or a
 * list of those; JSON arrays of scalars map to an unmodifiable list and are never nested. The object and its lists
 * are read-only, so one instance can be shared by every factory of a request. {@link #of} rejects a null value, a
 * blank name and any other value type by name, naming the argument as {@code vllm_xargs.<name>}; the typed getters
 * reject a missing argument or one of another type the same way.
 */
public final class LogitsProcessorArgs {

    private static final LogitsProcessorArgs EMPTY = new LogitsProcessorArgs(Collections.emptyMap());

    private final Map<String, Object> values;

    private LogitsProcessorArgs(Map<String, Object> values) {
        this.values = values;
    }

    /** The no-argument instance. */
    public static LogitsProcessorArgs empty() {
        return EMPTY;
    }

    /**
     * The arguments of {@code values}, copied and type-checked.
     *
     * @throws NullPointerException     when {@code values} is null
     * @throws IllegalArgumentException naming {@code vllm_xargs.<name>} for a blank name, a null value or a value
     *                                  (or list entry) that is not a string, boolean, integer, number or a list of
     *                                  those
     */
    public static LogitsProcessorArgs of(Map<String, Object> values) {
        Objects.requireNonNull(values, "vllm_xargs");
        if (values.isEmpty()) {
            return EMPTY;
        }
        Map<String, Object> copy = new LinkedHashMap<>();
        for (Map.Entry<String, Object> e : values.entrySet()) {
            String name = e.getKey();
            if (name == null || name.trim().isEmpty()) {
                throw new IllegalArgumentException("vllm_xargs has an argument with a blank name");
            }
            copy.put(name, value(name, e.getValue()));
        }
        return new LogitsProcessorArgs(Collections.unmodifiableMap(copy));
    }

    /** The normalized value of one argument, rejecting the types a JSON object can hold beyond the scalars. */
    private static Object value(String name, Object value) {
        String where = "vllm_xargs." + name;
        if (value == null) {
            throw new IllegalArgumentException(where + " must be a string, boolean, integer, number or an array"
                    + " of those, got null");
        }
        if (value instanceof String || value instanceof Boolean) {
            return value;
        }
        if (value instanceof Integer || value instanceof Long) {
            return ((Number) value).longValue();
        }
        if (value instanceof Float || value instanceof Double) {
            return ((Number) value).doubleValue();
        }
        if (value instanceof List) {
            List<?> list = (List<?>) value;
            List<Object> out = new ArrayList<>(list.size());
            for (Object entry : list) {
                if (entry == null) {
                    throw new IllegalArgumentException(where + "[" + out.size() + "] must not be null");
                }
                out.add(scalar(where + "[" + out.size() + "]", entry));
            }
            return Collections.unmodifiableList(out);
        }
        throw new IllegalArgumentException(where + " must be a string, boolean, integer, number or an array of"
                + " those, got " + typeName(value));
    }

    /** One list entry: a scalar, never another list or object. */
    private static Object scalar(String where, Object entry) {
        if (entry instanceof String || entry instanceof Boolean) {
            return entry;
        }
        if (entry instanceof Integer || entry instanceof Long) {
            return ((Number) entry).longValue();
        }
        if (entry instanceof Float || entry instanceof Double) {
            return ((Number) entry).doubleValue();
        }
        throw new IllegalArgumentException(where + " must be a string, boolean, integer or number, got "
                + typeName(entry));
    }

    /** Whether the request set any argument. */
    public boolean isEmpty() {
        return values.isEmpty();
    }

    /** Whether the request set {@code name}. */
    public boolean has(String name) {
        return values.containsKey(Objects.requireNonNull(name, "name"));
    }

    /** The normalized value of {@code name}, or null when the request set none. */
    public Object get(String name) {
        return values.get(Objects.requireNonNull(name, "name"));
    }

    /** The arguments, unmodifiable and in the request's order. */
    public Map<String, Object> values() {
        return values;
    }

    /**
     * The string argument {@code name}.
     *
     * @throws IllegalArgumentException when it is absent or is not a string
     */
    public String string(String name) {
        return typed(name, String.class);
    }

    /**
     * The integer argument {@code name}, which must fit an {@code int}.
     *
     * @throws IllegalArgumentException when it is absent, is not an integer, or lies outside the {@code int} range
     */
    public int integer(String name) {
        long value = typedLong(name);
        if (value < Integer.MIN_VALUE || value > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(where(name) + " is outside the integer range, got " + value);
        }
        return (int) value;
    }

    /**
     * The numeric argument {@code name}; an integer argument reads as its double value.
     *
     * @throws IllegalArgumentException when it is absent or is not a number
     */
    public double number(String name) {
        Object value = require(name);
        if (value instanceof Long || value instanceof Double) {
            return ((Number) value).doubleValue();
        }
        throw mismatch(name, "a number", value);
    }

    /**
     * The boolean argument {@code name}.
     *
     * @throws IllegalArgumentException when it is absent or is not a boolean
     */
    public boolean flag(String name) {
        return typed(name, Boolean.class);
    }

    /**
     * The list argument {@code name}, unmodifiable.
     *
     * @throws IllegalArgumentException when it is absent or is not a list
     */
    @SuppressWarnings("unchecked")
    public List<Object> list(String name) {
        Object value = require(name);
        if (value instanceof List) {
            return (List<Object>) value;
        }
        throw mismatch(name, "an array", value);
    }

    private long typedLong(String name) {
        Object value = require(name);
        if (value instanceof Long) {
            return (Long) value;
        }
        throw mismatch(name, "an integer", value);
    }

    private <T> T typed(String name, Class<T> type) {
        Object value = require(name);
        if (type.isInstance(value)) {
            return type.cast(value);
        }
        throw mismatch(name, type == String.class ? "a string" : "a boolean", value);
    }

    private Object require(String name) {
        Object value = values.get(Objects.requireNonNull(name, "name"));
        if (value == null) {
            throw new IllegalArgumentException(where(name) + " is missing");
        }
        return value;
    }

    private IllegalArgumentException mismatch(String name, String expected, Object value) {
        return new IllegalArgumentException(where(name) + " must be " + expected + ", got " + typeName(value));
    }

    private static String where(String name) {
        return "vllm_xargs." + name;
    }

    private static String typeName(Object value) {
        if (value == null) {
            return "null";
        }
        if (value instanceof String) {
            return "a string";
        }
        if (value instanceof Boolean) {
            return "a boolean";
        }
        if (value instanceof Long || value instanceof Integer) {
            return "an integer";
        }
        if (value instanceof Double || value instanceof Float) {
            return "a number";
        }
        if (value instanceof List) {
            return "a list";
        }
        if (value instanceof Map) {
            return "an object";
        }
        return value.getClass().getSimpleName();
    }
}

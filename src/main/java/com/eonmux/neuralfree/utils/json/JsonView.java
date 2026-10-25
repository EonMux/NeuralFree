package com.eonmux.neuralfree.utils.json;

import java.math.BigInteger;
import java.util.AbstractList;
import java.util.AbstractMap;
import java.util.AbstractSet;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.RandomAccess;
import java.util.Set;

/**
 * Immutable typed access to one JSON object of a {@link Json#parse parsed} document.
 *
 * <p>Every failure is a {@link JsonException} whose message names the document and the member, as
 * {@code "<source>:<path>: expected <type>, got <type>"} with paths such as {@code text_config.rope_scaling.type}
 * or {@code added_tokens[3].content}. Required accessors fail on a missing member; {@code opt*} accessors return
 * their default when the member is absent or {@code null}. Integer accessors accept integers and integral floats
 * within their range and reject fractions, booleans and out-of-range values. Containers handed out are read-only
 * views at every depth; the view never modifies the parsed value.</p>
 */
public final class JsonView {

    private final String source;
    private final String path;
    private final Map<String, Object> map;

    private JsonView(String source, String path, Map<String, Object> map) {
        this.source = source;
        this.path = path;
        this.map = map;
    }

    /**
     * A view of the root of {@code parsed}, a value returned by {@link Json#parse}.
     *
     * @param source names the document in messages (usually its file name)
     * @throws JsonException when {@code parsed} is not an object
     */
    public static JsonView of(Object parsed, String source) {
        return of(parsed, source, "");
    }

    /**
     * A view of {@code parsed} located at {@code path} of document {@code source}; {@code path} is empty for the root.
     *
     * @throws JsonException when {@code parsed} is not an object
     */
    @SuppressWarnings("unchecked")
    public static JsonView of(Object parsed, String source, String path) {
        Objects.requireNonNull(source, "source");
        Objects.requireNonNull(path, "path");
        if (!(parsed instanceof Map)) {
            throw new JsonException(source + ":" + (path.isEmpty() ? "<root>" : path) + ": expected object, got "
                    + typeName(parsed));
        }
        return new JsonView(source, path, (Map<String, Object>) parsed);
    }

    /** An empty object view at {@code path}, as returned by {@link #optObj} for an absent member. */
    public static JsonView empty(String source, String path) {
        return new JsonView(Objects.requireNonNull(source, "source"), Objects.requireNonNull(path, "path"),
                Collections.emptyMap());
    }

    public String source() { return source; }

    /** Path of this object from the document root; empty for the root. */
    public String path() { return path; }

    public boolean isEmpty() { return map.isEmpty(); }

    /** Whether the member exists, including with a {@code null} value. */
    public boolean has(String key) { return map.containsKey(key); }

    /** Member names in document order. */
    public Set<String> keys() { return Collections.unmodifiableSet(map.keySet()); }

    /** The member value as a read-only JSON value, or null when absent. */
    public Object raw(String key) { return readOnly(map.get(key)); }

    /** {@code "<source>:<path>.<key>"}, the location prefix of messages about member {@code key}. */
    public String where(String key) { return source + ":" + childPath(key); }

    // ------------------------------------------------------------------ objects

    public JsonView obj(String key) {
        Object v = require(key);
        if (!(v instanceof Map)) throw mismatch(key, "object", v);
        return of(v, source, childPath(key));
    }

    /** The member object, or an empty view at its path when it is absent or null. */
    public JsonView optObj(String key) {
        Object v = map.get(key);
        return v == null ? empty(source, childPath(key)) : obj(key);
    }

    // ------------------------------------------------------------------ scalars

    public String str(String key) { return asString(key, require(key)); }

    public String optStr(String key, String dflt) {
        Object v = map.get(key);
        return v == null ? dflt : asString(key, v);
    }

    public boolean bool(String key) { return asBoolean(key, require(key)); }

    public boolean optBool(String key, boolean dflt) {
        Object v = map.get(key);
        return v == null ? dflt : asBoolean(key, v);
    }

    public int i32(String key) { return asInt(childPath(key), require(key)); }

    public int optI32(String key, int dflt) {
        Object v = map.get(key);
        return v == null ? dflt : asInt(childPath(key), v);
    }

    public long i64(String key) { return asLong(childPath(key), require(key)); }

    public long optI64(String key, long dflt) {
        Object v = map.get(key);
        return v == null ? dflt : asLong(childPath(key), v);
    }

    /**
     * An integer clamped to the {@code long} range, or {@code dflt} when absent or null. For limits that configs
     * spell as a huge integer, such as transformers' {@code VERY_LARGE_INTEGER = int(1e30)} {@code model_max_length}.
     */
    public long i64Saturating(String key, long dflt) {
        Object v = map.get(key);
        if (v == null) return dflt;
        if (isOutOfLongRange(v)) return ((Number) v).doubleValue() > 0 ? Long.MAX_VALUE : Long.MIN_VALUE;
        return asLong(childPath(key), v);
    }

    /** A number as a double; integers convert as Python {@code float(int)}. */
    public double f64(String key) { return asDouble(key, require(key)); }

    public double optF64(String key, double dflt) {
        Object v = map.get(key);
        return v == null ? dflt : asDouble(key, v);
    }

    // ------------------------------------------------------------------ arrays

    /** The member array as a read-only list. */
    public List<Object> list(String key) {
        Object v = require(key);
        if (!(v instanceof List)) throw mismatch(key, "array", v);
        return new ReadOnlyList((List<?>) v);
    }

    /** The member array of strings. */
    public List<String> strList(String key) {
        List<Object> items = list(key);
        List<String> out = new ArrayList<>(items.size());
        for (int i = 0; i < items.size(); i++) {
            Object item = items.get(i);
            if (!(item instanceof String)) throw elementMismatch(key, i, "string", item);
            out.add((String) item);
        }
        return Collections.unmodifiableList(out);
    }

    /** The member array of 64-bit integers. */
    public List<Long> i64List(String key) {
        List<Object> items = list(key);
        List<Long> out = new ArrayList<>(items.size());
        for (int i = 0; i < items.size(); i++) out.add(asLong(childPath(key) + "[" + i + "]", items.get(i)));
        return Collections.unmodifiableList(out);
    }

    // ------------------------------------------------------------------ conversions

    private String childPath(String key) {
        return path.isEmpty() ? key : path + "." + key;
    }

    /** The member value, possibly null; a missing member is an error. */
    private Object require(String key) {
        Object v = map.get(key);
        if (v == null && !map.containsKey(key)) throw new JsonException(where(key) + ": required member is missing");
        return v;
    }

    private JsonException mismatch(String key, String expected, Object got) {
        return new JsonException(where(key) + ": expected " + expected + ", got " + typeName(got));
    }

    private JsonException elementMismatch(String key, int index, String expected, Object got) {
        return new JsonException(source + ":" + childPath(key) + "[" + index + "]: expected " + expected + ", got "
                + typeName(got));
    }

    private String asString(String key, Object v) {
        if (!(v instanceof String)) throw mismatch(key, "string", v);
        return (String) v;
    }

    private boolean asBoolean(String key, Object v) {
        if (!(v instanceof Boolean)) throw mismatch(key, "boolean", v);
        return (Boolean) v;
    }

    private double asDouble(String key, Object v) {
        if (v instanceof Double || v instanceof Long || v instanceof BigInteger) return ((Number) v).doubleValue();
        throw mismatch(key, "number", v);
    }

    private int asInt(String at, Object v) {
        if (isOutOfLongRange(v)) throw outOfRange(at, "int32", v);
        long l = asLong(at, v);
        if (l < Integer.MIN_VALUE || l > Integer.MAX_VALUE) throw outOfRange(at, "int32", v);
        return (int) l;
    }

    private long asLong(String at, Object v) {
        if (v instanceof Long) return (Long) v;
        if (isOutOfLongRange(v)) throw outOfRange(at, "int64", v);
        if (v instanceof BigInteger) return ((BigInteger) v).longValue();
        if (v instanceof Double) {
            double d = (Double) v;
            if (!isIntegral(d)) {
                throw new JsonException(source + ":" + at + ": expected integer, got number " + Json.floatRepr(d));
            }
            return (long) d;
        }
        throw new JsonException(source + ":" + at + ": expected integer, got " + typeName(v));
    }

    /** An integer or integral double outside [-2^63, 2^63). */
    private static boolean isOutOfLongRange(Object v) {
        if (v instanceof BigInteger) return ((BigInteger) v).bitLength() >= Long.SIZE;
        if (!(v instanceof Double)) return false;
        double d = (Double) v;
        return isIntegral(d) && (d < -0x1p63 || d >= 0x1p63);
    }

    private JsonException outOfRange(String at, String expected, Object v) {
        String text = v instanceof Double ? Json.floatRepr((Double) v) : String.valueOf(v);
        return new JsonException(source + ":" + at + ": expected " + expected + ", got " + typeName(v) + " " + text
                + " (out of range)");
    }

    private static boolean isIntegral(double d) {
        return !Double.isInfinite(d) && !Double.isNaN(d) && d == Math.rint(d);
    }

    /** JSON type name of a parsed value. */
    static String typeName(Object v) {
        if (v == null) return "null";
        if (v instanceof Boolean) return "boolean";
        if (v instanceof Long || v instanceof BigInteger) return "integer";
        if (v instanceof Double) return "number";
        if (v instanceof String) return "string";
        if (v instanceof List) return "array";
        if (v instanceof Map) return "object";
        return v.getClass().getSimpleName();
    }

    @Override
    public String toString() {
        return "JsonView{" + source + ":" + (path.isEmpty() ? "<root>" : path) + ", " + map.size() + " members}";
    }

    // ------------------------------------------------------------------ read-only containers

    private static Object readOnly(Object v) {
        if (v instanceof ReadOnlyList || v instanceof ReadOnlyMap) return v;
        if (v instanceof List) return new ReadOnlyList((List<?>) v);
        if (v instanceof Map) return new ReadOnlyMap((Map<?, ?>) v);
        return v;
    }

    /** Read-only list whose nested containers are read-only too. */
    private static final class ReadOnlyList extends AbstractList<Object> implements RandomAccess {
        private final List<?> items;

        ReadOnlyList(List<?> items) { this.items = items; }

        @Override public Object get(int index) { return readOnly(items.get(index)); }

        @Override public int size() { return items.size(); }
    }

    /** Read-only map whose nested containers are read-only too; keys keep document order. */
    private static final class ReadOnlyMap extends AbstractMap<Object, Object> {
        private final Map<?, ?> members;

        ReadOnlyMap(Map<?, ?> members) { this.members = members; }

        @Override public Object get(Object key) { return readOnly(members.get(key)); }

        @Override public boolean containsKey(Object key) { return members.containsKey(key); }

        @Override public int size() { return members.size(); }

        @Override
        public Set<Entry<Object, Object>> entrySet() {
            return new AbstractSet<Entry<Object, Object>>() {
                @Override public int size() { return members.size(); }

                @Override
                public Iterator<Entry<Object, Object>> iterator() {
                    Iterator<? extends Entry<?, ?>> it = members.entrySet().iterator();
                    return new Iterator<Entry<Object, Object>>() {
                        @Override public boolean hasNext() { return it.hasNext(); }

                        @Override
                        public Entry<Object, Object> next() {
                            Entry<?, ?> e = it.next();
                            return new SimpleImmutableEntry<>(e.getKey(), readOnly(e.getValue()));
                        }
                    };
                }
            };
        }
    }
}

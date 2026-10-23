package com.eonmux.neuralfree.gguf.hparams;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFValueType;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Typed, read-only view of GGUF metadata with llama.cpp loader semantics
 * (src/llama-model-loader.cpp {@code get_key}, {@code get_arr}, {@code get_key_or_arr}).
 *
 * <p>Every method takes a key from {@link GgufKeys}; a {@code %s} placeholder is replaced by the
 * {@code general.architecture} name. Optional getters return the supplied default only when the key is
 * absent; a present value of the wrong type or range throws {@link GgufFormatException} naming the key.
 * UINT32 values are read unsigned.
 */
public final class GgufKv {

    private final String archName;
    private final Map<String, Object> values;
    private final Map<String, GGUFValueType> types;
    private final Map<String, GGUFValueType> elementTypes;

    private GgufKv(Map<String, Object> values, Map<String, GGUFValueType> types,
                   Map<String, GGUFValueType> elementTypes) {
        this.values = values;
        this.types = types;
        this.elementTypes = elementTypes;
        Object arch = values.get(GgufKeys.GENERAL_ARCHITECTURE);
        this.archName = arch instanceof String ? (String) arch : null;
    }

    /** View over a parsed file, using the value types recorded by the parser. */
    public static GgufKv of(GGUFFileParser parser) {
        Map<String, Object> meta = parser.getMetadata();
        Map<String, GGUFValueType> t = new HashMap<>();
        Map<String, GGUFValueType> et = new HashMap<>();
        for (String key : meta.keySet()) {
            t.put(key, parser.getValueType(key));
            GGUFValueType e = parser.getArrayElementType(key);
            if (e != null) {
                et.put(key, e);
            }
        }
        return new GgufKv(Collections.unmodifiableMap(new HashMap<>(meta)), t, et);
    }

    /**
     * View over plain Java values. Types are inferred: Integer/Short/Byte as INT32, Long as INT64, Float as
     * FLOAT32, Double as FLOAT64, Boolean as BOOL, String as STRING; arrays are a {@link List} of such values
     * or a primitive/String Java array. Integer types are accepted wherever their value is in range, so
     * unsigned 32-bit values above 2^31 are given as Long.
     *
     * @throws IllegalArgumentException for an unsupported value class or a mixed-type list
     */
    public static GgufKv ofMap(Map<String, Object> map) {
        Map<String, Object> v = new HashMap<>();
        Map<String, GGUFValueType> t = new HashMap<>();
        Map<String, GGUFValueType> et = new HashMap<>();
        for (Map.Entry<String, Object> e : map.entrySet()) {
            String key = e.getKey();
            Object value = e.getValue();
            if (value == null) {
                throw new IllegalArgumentException("metadata key '" + key + "' has a null value");
            }
            List<Object> list = asList(value);
            if (list != null) {
                v.put(key, list);
                t.put(key, GGUFValueType.ARRAY);
                GGUFValueType elem = inferElementType(key, list);
                if (elem != null) {
                    et.put(key, elem);
                }
            } else {
                v.put(key, value);
                t.put(key, inferScalarType(key, value));
            }
        }
        return new GgufKv(Collections.unmodifiableMap(v), t, et);
    }

    /** The {@code general.architecture} value, or null when absent. */
    public String archName() {
        return archName;
    }

    /**
     * Concrete key for a {@link GgufKeys} constant.
     *
     * @throws GgufFormatException when the key is arch-scoped and the metadata has no architecture name
     */
    public String key(String keyFmt) {
        if (!keyFmt.contains("%s")) {
            return keyFmt;
        }
        if (archName == null) {
            throw new GgufFormatException("metadata has no general.architecture, cannot resolve key " + keyFmt);
        }
        return String.format(Locale.ROOT, keyFmt, archName);
    }

    /** True when the key is present (scalar or array). */
    public boolean has(String keyFmt) {
        return values.containsKey(key(keyFmt));
    }

    /** True when the key is present and holds an array. */
    public boolean isArray(String keyFmt) {
        return types.get(key(keyFmt)) == GGUFValueType.ARRAY;
    }

    /**
     * Stored value type of the key ({@link GGUFValueType#ARRAY} for an array), or null when the key is absent;
     * the type llama.cpp {@code gguf_get_kv_type} reports and {@code gguf_kv_to_str} formats by.
     */
    public GGUFValueType typeOf(String keyFmt) {
        return types.get(key(keyFmt));
    }

    /** Length of an array value, or -1 when the key is absent. */
    public int arrayLength(String keyFmt) {
        String k = key(keyFmt);
        Object v = values.get(k);
        if (v == null) {
            return -1;
        }
        return requireList(k, v).size();
    }

    // ---- scalars ----------------------------------------------------------------------------------------

    /** Unsigned 32-bit value that must fit an int, or {@code dflt} when absent. */
    public int u32(String keyFmt, int dflt) {
        String k = key(keyFmt);
        Object v = values.get(k);
        return v == null ? dflt : toInt(k, u32Value(k, v));
    }

    /** Required unsigned 32-bit value that must fit an int. */
    public int u32Required(String keyFmt) {
        String k = key(keyFmt);
        return toInt(k, u32Value(k, require(k)));
    }

    /** Unsigned 32-bit value over the full range {@code [0, 2^32 - 1]}, or {@code dflt} when absent. */
    public long u32Long(String keyFmt, long dflt) {
        String k = key(keyFmt);
        Object v = values.get(k);
        return v == null ? dflt : u32Value(k, v);
    }

    /** Signed integer value that must fit an int, or {@code dflt} when absent. */
    public int i32(String keyFmt, int dflt) {
        String k = key(keyFmt);
        Object v = values.get(k);
        if (v == null) {
            return dflt;
        }
        try {
            return GGUFValueType.int32Value(k, v, scalarType(k));
        } catch (IllegalArgumentException e) {
            throw wrap(e);
        }
    }

    /** Integer value (UINT64 must fit a signed long), or {@code dflt} when absent. */
    public long u64(String keyFmt, long dflt) {
        String k = key(keyFmt);
        Object v = values.get(k);
        if (v == null) {
            return dflt;
        }
        try {
            long x = GGUFValueType.integerValue(k, v, scalarType(k));
            if (x < 0) {
                throw new GgufFormatException("metadata key '" + k + "' value " + x + " is negative");
            }
            return x;
        } catch (IllegalArgumentException e) {
            throw wrap(e);
        }
    }

    /** Numeric value as float, or {@code dflt} when absent. */
    public float f32(String keyFmt, float dflt) {
        String k = key(keyFmt);
        Object v = values.get(k);
        return v == null ? dflt : f32Value(k, v);
    }

    /** Required numeric value as float. */
    public float f32Required(String keyFmt) {
        String k = key(keyFmt);
        return f32Value(k, require(k));
    }

    /** Boolean value, or {@code dflt} when absent. */
    public boolean bool(String keyFmt, boolean dflt) {
        String k = key(keyFmt);
        Object v = values.get(k);
        if (v == null) {
            return dflt;
        }
        try {
            return GGUFValueType.boolValue(k, v, scalarType(k));
        } catch (IllegalArgumentException e) {
            throw wrap(e);
        }
    }

    /** String value, or {@code dflt} when absent. */
    public String str(String keyFmt, String dflt) {
        String k = key(keyFmt);
        Object v = values.get(k);
        if (v == null) {
            return dflt;
        }
        try {
            return GGUFValueType.stringValue(k, v, scalarType(k));
        } catch (IllegalArgumentException e) {
            throw wrap(e);
        }
    }

    /** Required string value. */
    public String strRequired(String keyFmt) {
        String k = key(keyFmt);
        Object v = require(k);
        try {
            return GGUFValueType.stringValue(k, v, scalarType(k));
        } catch (IllegalArgumentException e) {
            throw wrap(e);
        }
    }

    // ---- arrays (null when absent) ------------------------------------------------------------------------

    /** Unsigned 32-bit array whose elements fit an int (BOOL elements as 0/1), or null when absent. */
    public int[] u32Array(String keyFmt) {
        String k = key(keyFmt);
        Object v = values.get(k);
        if (v == null) {
            return null;
        }
        try {
            long[] wide = GGUFValueType.uint32Array(k, requireList(k, v), elementTypes.get(k));
            int[] out = new int[wide.length];
            for (int i = 0; i < wide.length; i++) {
                out[i] = toInt(k + "[" + i + "]", wide[i]);
            }
            return out;
        } catch (IllegalArgumentException e) {
            throw wrap(e);
        }
    }

    /** Signed 32-bit array, or null when absent. */
    public int[] i32Array(String keyFmt) {
        String k = key(keyFmt);
        Object v = values.get(k);
        if (v == null) {
            return null;
        }
        try {
            return GGUFValueType.int32Array(k, requireList(k, v), elementTypes.get(k));
        } catch (IllegalArgumentException e) {
            throw wrap(e);
        }
    }

    /** Unsigned 64-bit array (UINT64 elements keep their bit pattern), or null when absent. */
    public long[] u64Array(String keyFmt) {
        String k = key(keyFmt);
        Object v = values.get(k);
        if (v == null) {
            return null;
        }
        try {
            return GGUFValueType.uint64Array(k, requireList(k, v), elementTypes.get(k));
        } catch (IllegalArgumentException e) {
            throw wrap(e);
        }
    }

    /** Numeric array as floats, or null when absent. */
    public float[] f32Array(String keyFmt) {
        String k = key(keyFmt);
        Object v = values.get(k);
        if (v == null) {
            return null;
        }
        try {
            return GGUFValueType.floatArray(k, requireList(k, v), elementTypes.get(k));
        } catch (IllegalArgumentException e) {
            throw wrap(e);
        }
    }

    /** BOOL or integer array as booleans, or null when absent. */
    public boolean[] boolArray(String keyFmt) {
        String k = key(keyFmt);
        Object v = values.get(k);
        if (v == null) {
            return null;
        }
        try {
            return GGUFValueType.boolArray(k, requireList(k, v), elementTypes.get(k));
        } catch (IllegalArgumentException e) {
            throw wrap(e);
        }
    }

    /** STRING array, or null when absent. */
    public String[] strArray(String keyFmt) {
        String k = key(keyFmt);
        Object v = values.get(k);
        if (v == null) {
            return null;
        }
        try {
            return GGUFValueType.stringArray(k, requireList(k, v), elementTypes.get(k));
        } catch (IllegalArgumentException e) {
            throw wrap(e);
        }
    }

    // ---- per-layer values (llama.cpp get_key_or_arr, src/llama-model-loader.cpp:460-498) -----------------

    /** Per-layer unsigned values: {@code dflt} for every layer when absent. See {@link #keyOrArr(String, int, int[])}. */
    public int[] keyOrArr(String keyFmt, int n, int dflt) {
        int[] fill = new int[n];
        Arrays.fill(fill, dflt);
        return keyOrArr(keyFmt, n, fill);
    }

    /**
     * Per-layer unsigned values: an absent key yields a copy of {@code fallback}, a scalar is broadcast to
     * {@code n} entries, and an array must have exactly {@code n} elements.
     *
     * @throws GgufFormatException when the array length differs from {@code n} or an element is malformed
     */
    public int[] keyOrArr(String keyFmt, int n, int[] fallback) {
        String k = key(keyFmt);
        Object v = values.get(k);
        if (v == null) {
            return copyOf(k, fallback, n);
        }
        if (types.get(k) == GGUFValueType.ARRAY) {
            checkLength(k, v, n);
            return u32Array(keyFmt);
        }
        int[] out = new int[n];
        Arrays.fill(out, toInt(k, u32Value(k, v)));
        return out;
    }

    /** Float variant of {@link #keyOrArr(String, int, int[])}; absent yields {@code dflt} for every layer. */
    public float[] keyOrArrF32(String keyFmt, int n, float dflt) {
        String k = key(keyFmt);
        Object v = values.get(k);
        float[] out = new float[n];
        if (v == null) {
            Arrays.fill(out, dflt);
            return out;
        }
        if (types.get(k) == GGUFValueType.ARRAY) {
            checkLength(k, v, n);
            return f32Array(keyFmt);
        }
        Arrays.fill(out, f32Value(k, v));
        return out;
    }

    // ---- internals --------------------------------------------------------------------------------------

    private Object require(String k) {
        Object v = values.get(k);
        if (v == null) {
            throw new GgufFormatException("required metadata key '" + k + "' is missing");
        }
        return v;
    }

    private GGUFValueType scalarType(String k) {
        GGUFValueType t = types.get(k);
        if (t == GGUFValueType.ARRAY) {
            throw new GgufFormatException("metadata key '" + k + "' must be a scalar but is an array");
        }
        return t;
    }

    private long u32Value(String k, Object v) {
        try {
            return GGUFValueType.uint32Value(k, v, scalarType(k));
        } catch (IllegalArgumentException e) {
            throw wrap(e);
        }
    }

    private float f32Value(String k, Object v) {
        try {
            return GGUFValueType.floatValue(k, v, scalarType(k));
        } catch (IllegalArgumentException e) {
            throw wrap(e);
        }
    }

    private static int toInt(String k, long v) {
        if (v > Integer.MAX_VALUE) {
            throw new GgufFormatException("metadata key '" + k + "' value " + v + " exceeds the supported maximum "
                    + Integer.MAX_VALUE);
        }
        return (int) v;
    }

    private void checkLength(String k, Object v, int n) {
        int len = requireList(k, v).size();
        if (len != n) {
            throw new GgufFormatException("metadata key '" + k + "' has wrong array length; expected " + n
                    + ", got " + len);
        }
    }

    @SuppressWarnings("unchecked")
    private List<Object> requireList(String k, Object v) {
        if (!(v instanceof List)) {
            throw new GgufFormatException("metadata key '" + k + "' must be an array but has GGUF type " + types.get(k));
        }
        return (List<Object>) v;
    }

    private static int[] copyOf(String k, int[] fallback, int n) {
        if (fallback.length != n) {
            throw new IllegalArgumentException("fallback for '" + k + "' has length " + fallback.length + ", expected " + n);
        }
        return fallback.clone();
    }

    private static GgufFormatException wrap(IllegalArgumentException e) {
        if (e instanceof GgufFormatException) {
            return (GgufFormatException) e;
        }
        return new GgufFormatException(e.getMessage(), e);
    }

    private static List<Object> asList(Object value) {
        if (value instanceof List) {
            return new ArrayList<>((List<?>) value);
        }
        if (value.getClass().isArray()) {
            int n = Array.getLength(value);
            List<Object> out = new ArrayList<>(n);
            for (int i = 0; i < n; i++) {
                out.add(Array.get(value, i));
            }
            return out;
        }
        return null;
    }

    private static GGUFValueType inferElementType(String key, List<Object> list) {
        GGUFValueType elem = null;
        for (int i = 0; i < list.size(); i++) {
            Object o = list.get(i);
            if (o == null) {
                throw new IllegalArgumentException("metadata key '" + key + "[" + i + "]' is null");
            }
            if (o instanceof Short || o instanceof Byte) {
                list.set(i, ((Number) o).intValue());
            }
            GGUFValueType t = inferScalarType(key, o);
            if (elem != null && elem != t) {
                throw new IllegalArgumentException("metadata key '" + key + "' mixes " + elem + " and " + t + " elements");
            }
            elem = t;
        }
        return elem;
    }

    private static GGUFValueType inferScalarType(String key, Object v) {
        if (v instanceof Integer || v instanceof Short || v instanceof Byte) {
            return GGUFValueType.INT32;
        }
        if (v instanceof Long) {
            return GGUFValueType.INT64;
        }
        if (v instanceof Float) {
            return GGUFValueType.FLOAT32;
        }
        if (v instanceof Double) {
            return GGUFValueType.FLOAT64;
        }
        if (v instanceof Boolean) {
            return GGUFValueType.BOOL;
        }
        if (v instanceof String) {
            return GGUFValueType.STRING;
        }
        throw new IllegalArgumentException("metadata key '" + key + "' has unsupported value class "
                + v.getClass().getName());
    }
}

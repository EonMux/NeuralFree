package com.eonmux.neuralfree.gguf;

import java.util.List;

/**
 * GGUF metadata value types (ggml/include/gguf.h {@code gguf_type}) and checked conversions of the Java
 * values {@link GGUFFileParser} decodes for them.
 *
 * <p>Decoded representations: 8/16/32-bit integers are {@link Integer} (UINT32 keeps its bit pattern, so
 * values of 2^31 and above are negative until read through an unsigned conversion), 64-bit integers are
 * {@link Long} (UINT64 keeps its bit pattern), FLOAT32 is {@link Float}, FLOAT64 is {@link Double}, BOOL is
 * {@link Boolean}, STRING is {@link String} and ARRAY is a {@code List<Object>} of element values.
 *
 * <p>Every conversion names the metadata key in its {@link IllegalArgumentException} so malformed
 * metadata is reported instead of being replaced by a default.
 */
public enum GGUFValueType {
    UINT8(0),
    INT8(1),
    UINT16(2),
    INT16(3),
    UINT32(4),
    INT32(5),
    FLOAT32(6),
    BOOL(7),
    STRING(8),
    ARRAY(9),
    UINT64(10),
    INT64(11),
    FLOAT64(12);

    private static final long UINT32_MAX = 0xFFFF_FFFFL;
    private static final GGUFValueType[] BY_ID = values();

    private final int id;

    GGUFValueType(int id) {
        this.id = id;
    }

    /** The type id stored in the file. */
    public int id() {
        return id;
    }

    /** True for the eight integer types. */
    public boolean isInteger() {
        switch (this) {
            case UINT8: case INT8: case UINT16: case INT16:
            case UINT32: case INT32: case UINT64: case INT64:
                return true;
            default:
                return false;
        }
    }

    /** True for FLOAT32 and FLOAT64. */
    public boolean isFloat() {
        return this == FLOAT32 || this == FLOAT64;
    }

    /**
     * Type for a file type id.
     *
     * @throws IllegalArgumentException for an id outside 0..12
     */
    public static GGUFValueType fromId(int id) {
        if (id < 0 || id >= BY_ID.length) {
            throw new IllegalArgumentException("unknown GGUF value type id " + id);
        }
        return BY_ID[id];
    }

    /**
     * Exact integer value of an integer-typed metadata value: UINT32 is widened unsigned, UINT64 must fit
     * a signed long.
     */
    public static long integerValue(String key, Object value, GGUFValueType type) {
        if (type == null || !type.isInteger() || !(value instanceof Number)) {
            throw mismatch(key, "an integer", type);
        }
        Number n = (Number) value;
        if (type == UINT32) {
            return Integer.toUnsignedLong(n.intValue());
        }
        long v = n.longValue();
        if (type == UINT64 && v < 0) {
            throw new IllegalArgumentException("metadata key '" + key + "' value "
                    + Long.toUnsignedString(v) + " exceeds the signed 64-bit range");
        }
        return v;
    }

    /** Value of an integer-typed metadata value that must lie in {@code [0, 2^32 - 1]}. */
    public static long uint32Value(String key, Object value, GGUFValueType type) {
        long v = integerValue(key, value, type);
        if (v < 0 || v > UINT32_MAX) {
            throw new IllegalArgumentException("metadata key '" + key + "' value " + v + " is outside the uint32 range");
        }
        return v;
    }

    /** Value of an integer-typed metadata value that must fit a Java int. */
    public static int int32Value(String key, Object value, GGUFValueType type) {
        long v = integerValue(key, value, type);
        if (v < Integer.MIN_VALUE || v > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("metadata key '" + key + "' value " + v + " is outside the int32 range");
        }
        return (int) v;
    }

    /** Numeric metadata value as a float; integer types are converted exactly as a long first. */
    public static float floatValue(String key, Object value, GGUFValueType type) {
        if (type != null && type.isFloat() && value instanceof Number) {
            return ((Number) value).floatValue();
        }
        if (type != null && type.isInteger()) {
            return (float) integerValue(key, value, type);
        }
        throw mismatch(key, "a number", type);
    }

    /** Boolean metadata value; integer types are true when non-zero (llama.cpp array BOOL handling). */
    public static boolean boolValue(String key, Object value, GGUFValueType type) {
        if (type == BOOL && value instanceof Boolean) {
            return (Boolean) value;
        }
        if (type != null && type.isInteger()) {
            return integerValue(key, value, type) != 0;
        }
        throw mismatch(key, "a bool", type);
    }

    /** String metadata value. */
    public static String stringValue(String key, Object value, GGUFValueType type) {
        if (type == STRING && value instanceof String) {
            return (String) value;
        }
        throw mismatch(key, "a string", type);
    }

    /**
     * Elements of an unsigned 32-bit array (UINT8..UINT32, non-negative signed integers and BOOL).
     *
     * @param elementType element type; may be null only for an empty array
     */
    public static long[] uint32Array(String key, List<?> values, GGUFValueType elementType) {
        long[] out = new long[values.size()];
        for (int i = 0; i < out.length; i++) {
            Object v = values.get(i);
            out[i] = elementType == BOOL ? (boolValue(element(key, i), v, BOOL) ? 1 : 0)
                    : uint32Value(element(key, i), v, elementType);
        }
        return out;
    }

    /** Elements of a signed 32-bit array (any integer type with in-range values, and BOOL). */
    public static int[] int32Array(String key, List<?> values, GGUFValueType elementType) {
        int[] out = new int[values.size()];
        for (int i = 0; i < out.length; i++) {
            Object v = values.get(i);
            out[i] = elementType == BOOL ? (boolValue(element(key, i), v, BOOL) ? 1 : 0)
                    : int32Value(element(key, i), v, elementType);
        }
        return out;
    }

    /**
     * Elements of a 64-bit integer array. UINT64 elements keep their bit pattern (read them with
     * {@link Long#toUnsignedString(long)} semantics); other integer types are widened exactly.
     */
    public static long[] uint64Array(String key, List<?> values, GGUFValueType elementType) {
        long[] out = new long[values.size()];
        for (int i = 0; i < out.length; i++) {
            Object v = values.get(i);
            if (elementType == UINT64 && v instanceof Number) {
                out[i] = ((Number) v).longValue();
            } else {
                long x = integerValue(element(key, i), v, elementType);
                if (x < 0) {
                    throw new IllegalArgumentException("metadata key '" + element(key, i) + "' value " + x
                            + " is negative in an unsigned 64-bit array");
                }
                out[i] = x;
            }
        }
        return out;
    }

    /** Elements of a numeric array as floats. */
    public static float[] floatArray(String key, List<?> values, GGUFValueType elementType) {
        float[] out = new float[values.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = floatValue(element(key, i), values.get(i), elementType);
        }
        return out;
    }

    /** Elements of a BOOL or integer array as booleans (non-zero is true). */
    public static boolean[] boolArray(String key, List<?> values, GGUFValueType elementType) {
        boolean[] out = new boolean[values.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = boolValue(element(key, i), values.get(i), elementType);
        }
        return out;
    }

    /** Elements of a STRING array. */
    public static String[] stringArray(String key, List<?> values, GGUFValueType elementType) {
        String[] out = new String[values.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = stringValue(element(key, i), values.get(i), elementType);
        }
        return out;
    }

    private static String element(String key, int index) {
        return key + "[" + index + "]";
    }

    private static IllegalArgumentException mismatch(String key, String expected, GGUFValueType actual) {
        return new IllegalArgumentException("metadata key '" + key + "' must be " + expected
                + " but has GGUF type " + (actual == null ? "unknown" : actual.name()));
    }
}

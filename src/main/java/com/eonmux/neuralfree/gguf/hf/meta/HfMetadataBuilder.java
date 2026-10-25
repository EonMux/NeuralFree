package com.eonmux.neuralfree.gguf.hf.meta;

import com.eonmux.neuralfree.gguf.GGUFValueType;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.lang.reflect.Array;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Collects the metadata of a Hugging Face checkpoint's GGUF view the way {@code GGUFWriter} does
 * (LC/gguf-py/gguf/gguf_writer.py): keys named by {@link com.eonmux.neuralfree.gguf.hparams.GgufKeys} templates, typed
 * by {@link GgufKeyTable}, values checked and converted to the {@link GGUFValueType} Java value model. Writing a key
 * again replaces its value and keeps its position, as {@code add_key_value} does (gguf_writer.py:278-282); an empty
 * array written through {@link #set} is skipped, as {@code add_array} does (gguf_writer.py:326-329).
 *
 * <p>Value conversion: integer types take any integral {@link Number} (integral doubles included) in the type's range
 * and reject fractions and booleans; UINT32 is stored with its bit pattern in an {@link Integer} and UINT64 in a
 * {@link Long}. FLOAT32 takes any number through a {@code (float)} cast and rejects NaN. BOOL takes a
 * {@link Boolean}, STRING a {@link String}. Arrays take a {@link List}, an object array or a primitive array whose
 * elements convert to the element type. A key whose writer takes an integer or a list
 * ({@link GgufKeyTable.KeyType#any()}) is UINT32 for a number and an array for a list, with the element type of the
 * values (integers INT32, other numbers FLOAT32, booleans BOOL, strings STRING; {@code GGUFValueType.get_type},
 * constants.py:6364-6377).</p>
 *
 * <p>Mutable and single-threaded; {@link #build()} returns an immutable snapshot.</p>
 */
public final class HfMetadataBuilder {

    private static final Logger LOG = LogManager.getLogger(HfMetadataBuilder.class);

    private static final String ARCH_PREFIX = "%s.";
    private static final long UINT32_LIMIT = 1L << 32;

    private final String arch;
    private final Map<String, Object> values = new LinkedHashMap<>();
    private final Map<String, GGUFValueType> types = new LinkedHashMap<>();
    private final Map<String, GGUFValueType> elementTypes = new LinkedHashMap<>();

    /**
     * A builder for a model of GGUF architecture {@code arch}.
     *
     * @throws IllegalArgumentException when the name is empty or holds {@code %}, {@code .} or whitespace
     */
    public HfMetadataBuilder(String arch) {
        Objects.requireNonNull(arch, "arch");
        if (arch.isEmpty() || !arch.matches("[A-Za-z0-9_\\-]+")) {
            throw new IllegalArgumentException("invalid GGUF architecture name '" + arch + "'");
        }
        this.arch = arch;
    }

    /** The architecture name that replaces a leading {@code %s.}. */
    public String arch() {
        return arch;
    }

    /**
     * The concrete key of {@code template}: a leading {@code %s.} becomes {@code <arch>.}; other text is kept.
     */
    public String key(String template) {
        Objects.requireNonNull(template, "template");
        return template.startsWith(ARCH_PREFIX) ? arch + template.substring(2) : template;
    }

    /**
     * Writes {@code value} under {@code template}, typed by {@link GgufKeyTable}. An empty array is not written.
     *
     * @throws IllegalArgumentException when the table has no such template, the template has index placeholders
     *                                  (use {@link #setAt}) or names a module family ({@code {dense}}) instead of
     *                                  one module, or the value does not convert to the key's type
     */
    public void set(String template, Object value) {
        String body = template.startsWith(ARCH_PREFIX) ? template.substring(ARCH_PREFIX.length()) : template;
        if (body.contains("%")) {
            throw new IllegalArgumentException("metadata key template '" + template
                    + "' has index placeholders; write it with setAt");
        }
        requireConcrete(template, body);
        write(key(template), requireType(template), value);
    }

    /**
     * Writes {@code value} under {@code template} with its placeholders after a leading {@code %s.} filled by
     * {@code parts} in order ({@code String.format}), typed by the template. An empty array is not written.
     *
     * @throws IllegalArgumentException when the table has no such template, the parts do not fit its placeholders or
     *                                  contain a {@code %}, or the value does not convert to the key's type
     */
    public void setAt(String template, Object value, Object... parts) {
        GgufKeyTable.KeyType type = requireType(template);
        boolean archScoped = template.startsWith(ARCH_PREFIX);
        String body = archScoped ? template.substring(ARCH_PREFIX.length()) : template;
        String filled;
        try {
            filled = String.format(Locale.ROOT, body, parts);
        } catch (java.util.IllegalFormatException e) {
            throw new IllegalArgumentException("parts " + java.util.Arrays.toString(parts)
                    + " do not fit metadata key template '" + template + "'", e);
        }
        if (filled.contains("%") || countPlaceholders(body) != parts.length) {
            throw new IllegalArgumentException("parts " + java.util.Arrays.toString(parts)
                    + " do not fit metadata key template '" + template + "'");
        }
        requireConcrete(template, body);
        write(archScoped ? arch + "." + filled : filled, type, value);
    }

    /**
     * Writes the scalar {@code value} under {@code key} (a leading {@code %s.} is the architecture) with an explicit
     * type, for keys the table does not type. Arrays use {@link #setTyped(String, GGUFValueType, GGUFValueType,
     * Object)}.
     *
     * @throws IllegalArgumentException when {@code type} is ARRAY or the value does not convert to it
     */
    public void setTyped(String key, GGUFValueType type, Object value) {
        if (type == GGUFValueType.ARRAY) {
            throw new IllegalArgumentException("'" + key + "': an array needs an element type");
        }
        setTyped(key, type, null, value);
    }

    /**
     * Writes {@code value} under {@code key} (a leading {@code %s.} is the architecture) with an explicit type and,
     * for an array, element type. An empty array is written.
     *
     * @throws IllegalArgumentException when the types do not fit together or the value does not convert to them
     */
    public void setTyped(String key, GGUFValueType type, GGUFValueType elementType, Object value) {
        Objects.requireNonNull(type, "type");
        String k = key(key);
        if ((type == GGUFValueType.ARRAY) != (elementType != null) || elementType == GGUFValueType.ARRAY) {
            throw new IllegalArgumentException("'" + k + "': type " + type + " with element type " + elementType);
        }
        Object v = type == GGUFValueType.ARRAY ? convertArray(k, elementType, value) : convertScalar(k, type, value);
        store(k, type, elementType, v);
    }

    /** True when the key of {@code template} (see {@link #key}) is written. */
    public boolean has(String template) {
        return values.containsKey(key(template));
    }

    /** The value of the key of {@code template}, or null when it is not written. */
    public Object get(String template) {
        return values.get(key(template));
    }

    /** The type of the key of {@code template}, or null when it is not written. */
    public GGUFValueType type(String template) {
        return types.get(key(template));
    }

    /** Removes the key of {@code template}; returns its value, or null when it was not written. */
    public Object remove(String template) {
        String k = key(template);
        types.remove(k);
        elementTypes.remove(k);
        return values.remove(k);
    }

    /**
     * Writes every entry of {@code kv}, whose keys are concrete ({@code tokenizer.ggml.tokens},
     * {@code <arch>.vocab_size}), each typed through {@link GgufKeyTable#resolve}.
     *
     * @throws IllegalArgumentException when a key has no type or a value does not convert to it
     */
    public void putAll(Map<String, ?> kv) {
        for (Map.Entry<String, ?> e : kv.entrySet()) {
            String k = Objects.requireNonNull(e.getKey(), "key");
            Map.Entry<String, GgufKeyTable.KeyType> t = GgufKeyTable.resolve(k, arch);
            if (t == null) {
                throw new IllegalArgumentException("metadata key '" + k + "' has no type in GgufKeyTable");
            }
            write(k, t.getValue(), e.getValue());
        }
    }

    /** An immutable snapshot of the written keys. */
    public HfMetadata build() {
        return new HfMetadata(values, types, elementTypes);
    }

    // ------------------------------------------------------------------ conversion

    private static GgufKeyTable.KeyType requireType(String template) {
        GgufKeyTable.KeyType t = GgufKeyTable.type(Objects.requireNonNull(template, "template"));
        if (t == null) {
            throw new IllegalArgumentException("metadata key template '" + template + "' has no type in GgufKeyTable");
        }
        return t;
    }

    /** Rejects a key that still holds a table placeholder such as {@code {dense}}. */
    private static void requireConcrete(String template, String body) {
        if (body.indexOf('{') >= 0) {
            throw new IllegalArgumentException("metadata key template '" + template
                    + "' names a module family; write the key of one module (for example dense_2)");
        }
    }

    private void write(String key, GgufKeyTable.KeyType t, Object value) {
        Objects.requireNonNull(value, () -> "metadata key '" + key + "' has a null value");
        GGUFValueType type = t.type();
        GGUFValueType elem = t.elementType();
        if (t.any()) {
            if (value instanceof Number) {
                type = GGUFValueType.UINT32;
                elem = null;
            } else {
                type = GGUFValueType.ARRAY;
                elem = inferElementType(key, value);
            }
        }
        if (type == GGUFValueType.ARRAY) {
            List<Object> list = convertArray(key, elem, value);
            if (list.isEmpty()) {
                return;
            }
            store(key, type, elem, list);
        } else {
            store(key, type, null, convertScalar(key, type, value));
        }
    }

    private void store(String key, GGUFValueType type, GGUFValueType elem, Object value) {
        if (values.containsKey(key)) {
            LOG.debug("metadata key '{}' written again, replacing {} with {}", key, values.get(key), value);
        }
        values.put(key, value);
        types.put(key, type);
        if (elem != null) {
            elementTypes.put(key, elem);
        } else {
            elementTypes.remove(key);
        }
    }

    /** The element type {@code GGUFValueType.get_type} gives the items of {@code value}, all of which must agree. */
    static GGUFValueType inferElementType(String key, Object value) {
        List<Object> items = asList(key, value);
        if (items.isEmpty()) {
            return GGUFValueType.INT32;
        }
        GGUFValueType first = null;
        for (Object item : items) {
            GGUFValueType t = pythonType(key, item);
            if (first == null) {
                first = t;
            } else if (t != first) {
                throw new IllegalArgumentException("metadata key '" + key
                        + "': all items of a GGUF array must have the same type, found " + first + " and " + t);
            }
        }
        return first;
    }

    private static GGUFValueType pythonType(String key, Object item) {
        if (item instanceof String) {
            return GGUFValueType.STRING;
        }
        if (item instanceof Boolean) {
            return GGUFValueType.BOOL;
        }
        if (item instanceof Float || item instanceof Double) {
            return GGUFValueType.FLOAT32;
        }
        if (item instanceof Integer || item instanceof Long || item instanceof Short || item instanceof Byte
                || item instanceof BigInteger) {
            return GGUFValueType.INT32;
        }
        throw new IllegalArgumentException("metadata key '" + key + "': unsupported array item "
                + (item == null ? "null" : item.getClass().getSimpleName()));
    }

    static List<Object> convertArray(String key, GGUFValueType elem, Object value) {
        List<Object> items = asList(key, value);
        List<Object> out = new ArrayList<>(items.size());
        for (int i = 0; i < items.size(); i++) {
            out.add(convertScalar(key + "[" + i + "]", elem, items.get(i)));
        }
        return Collections.unmodifiableList(out);
    }

    private static List<Object> asList(String key, Object value) {
        if (value instanceof List) {
            return new ArrayList<>((List<?>) value);
        }
        if (value != null && value.getClass().isArray()) {
            int n = Array.getLength(value);
            List<Object> out = new ArrayList<>(n);
            for (int i = 0; i < n; i++) {
                out.add(Array.get(value, i));
            }
            return out;
        }
        throw new IllegalArgumentException("metadata key '" + key + "' is an array, got "
                + (value == null ? "null" : value.getClass().getSimpleName()));
    }

    static Object convertScalar(String key, GGUFValueType type, Object value) {
        if (value == null) {
            throw new IllegalArgumentException("metadata key '" + key + "' has a null value");
        }
        switch (type) {
            case UINT8: return (int) integer(key, value, 0, 0xFF);
            case INT8: return (int) integer(key, value, Byte.MIN_VALUE, Byte.MAX_VALUE);
            case UINT16: return (int) integer(key, value, 0, 0xFFFF);
            case INT16: return (int) integer(key, value, Short.MIN_VALUE, Short.MAX_VALUE);
            case UINT32: return (int) integer(key, value, 0, UINT32_LIMIT - 1);
            case INT32: return (int) integer(key, value, Integer.MIN_VALUE, Integer.MAX_VALUE);
            case INT64: return integer(key, value, Long.MIN_VALUE, Long.MAX_VALUE);
            case UINT64: return uint64(key, value);
            case FLOAT32: return float32(key, value);
            case FLOAT64: return float64(key, value);
            case BOOL:
                if (!(value instanceof Boolean)) {
                    throw mismatch(key, type, value);
                }
                return value;
            case STRING:
                if (!(value instanceof String)) {
                    throw mismatch(key, type, value);
                }
                return value;
            default:
                throw new IllegalArgumentException("metadata key '" + key + "': " + type + " is not a scalar type");
        }
    }

    private static long integer(String key, Object value, long min, long max) {
        long v;
        if (value instanceof Integer || value instanceof Long || value instanceof Short || value instanceof Byte) {
            v = ((Number) value).longValue();
        } else if (value instanceof BigInteger && ((BigInteger) value).bitLength() < Long.SIZE) {
            v = ((BigInteger) value).longValue();
        } else if ((value instanceof Double || value instanceof Float) && isIntegral(((Number) value).doubleValue())) {
            v = (long) ((Number) value).doubleValue();
        } else {
            throw new IllegalArgumentException("metadata key '" + key + "' must be an integer, got "
                    + describe(value));
        }
        if (v < min || v > max) {
            throw new IllegalArgumentException("metadata key '" + key + "' value " + v + " is outside ["
                    + min + ", " + max + "]");
        }
        return v;
    }

    private static long uint64(String key, Object value) {
        if (value instanceof BigInteger) {
            BigInteger b = (BigInteger) value;
            if (b.signum() < 0 || b.bitLength() > Long.SIZE) {
                throw new IllegalArgumentException("metadata key '" + key + "' value " + b
                        + " is outside the UINT64 range");
            }
            return b.longValue();
        }
        return integer(key, value, 0, Long.MAX_VALUE);
    }

    private static boolean isIntegral(double d) {
        return d == Math.rint(d) && !Double.isInfinite(d) && d >= -0x1p63 && d < 0x1p63;
    }

    private static float float32(String key, Object value) {
        if (!(value instanceof Number) || value instanceof BigInteger && ((BigInteger) value).bitLength() > 1023) {
            throw mismatch(key, GGUFValueType.FLOAT32, value);
        }
        float f = (float) ((Number) value).doubleValue();
        if (Float.isNaN(f)) {
            throw new IllegalArgumentException("metadata key '" + key + "' is NaN");
        }
        return f;
    }

    private static double float64(String key, Object value) {
        if (!(value instanceof Number)) {
            throw mismatch(key, GGUFValueType.FLOAT64, value);
        }
        double d = ((Number) value).doubleValue();
        if (Double.isNaN(d)) {
            throw new IllegalArgumentException("metadata key '" + key + "' is NaN");
        }
        return d;
    }

    private static int countPlaceholders(String body) {
        int n = 0;
        for (int i = body.indexOf('%'); i >= 0; i = body.indexOf('%', i + 2)) {
            n++;
        }
        return n;
    }

    private static IllegalArgumentException mismatch(String key, GGUFValueType type, Object value) {
        return new IllegalArgumentException("metadata key '" + key + "' must be " + type + ", got "
                + describe(value));
    }

    private static String describe(Object value) {
        return value.getClass().getSimpleName() + " " + value;
    }
}

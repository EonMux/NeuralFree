package com.eonmux.neuralfree.gguf.load;

import com.eonmux.neuralfree.gguf.GGUFValueType;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * One tensor metadata override of the llama.cpp {@code --override-kv key=type:value} form
 * (src/llama-model-loader.cpp:185-249). Applied after the source is built, by the view of
 * {@link OverriddenSource}: it rewrites the metadata of the source's GGUF view for GGUF and Hugging Face
 * checkpoints alike, post-conversion. It is the post-hoc sibling of the pre-conversion
 * {@code neuralfree.hf.overrides} JSON ({@code --hf-overrides}): that one changes what the converter sees, this
 * one changes what every consumer of the opened source sees.
 *
 * <p>Types are llama.cpp's four flag types; each stores its GGUF metadata type as
 * {@code gguf_set_kv_*} writes it ({@code int} as INT32, {@code float} as FLOAT32):
 * <ul>
 *   <li>{@code str} &rarr; STRING, the rest of the value after {@code str:} with no leading or trailing
 *       spaces removed (a value may contain {@code :} and {@code =});</li>
 *   <li>{@code bool} &rarr; BOOL, {@code true} or {@code false};</li>
 *   <li>{@code int} &rarr; INT32, a signed 64-bit literal that must fit 32 bits (out-of-range values are
 *       rejected naming the entry instead of being truncated);</li>
 *   <li>{@code float} &rarr; FLOAT32, a finite decimal literal.</li>
 * </ul>
 * The wider scalar types of the GGUF store are accepted by name too, for the tools that edit a file's key/value
 * store: {@code uint8}, {@code int8}, {@code uint16}, {@code int16}, {@code uint32}, {@code int64} and
 * {@code float64}, each stored as the type its name says with its value range checked.
 * Parsers reject an entry without {@code type:value}, an unknown type, or a value of the wrong form, naming the
 * entry (and, in a CLI such as {@code LLMDemoOptions} with its own {@code --override-kv} flag, the flag).
 *
 * <p>The {@code --metadata} file of the converters ({@link ConverterMetadata}) builds overrides the
 * {@code key=type:value} grammar cannot carry: UINT32, INT64 and FLOAT64 scalars ({@link Override#scalar}) and
 * ARRAY values with an element type ({@link Override#array}). Those render informatively in
 * {@link Override#toString()} but do not round-trip through {@link #parse(String, String)}.</p>
 */
public final class MetadataOverrides {

    /** The property names of {@code --override-kv}, on the {@code neuralfree.*} property base of ModelSourceOptions. */
    public static final String PROP_OVERRIDE_KV = "neuralfree.override.kv";

    private MetadataOverrides() {
    }

    /** One {@code key=type:value} override; immutable and typed in the GGUF value model. */
    public static final class Override {

        private final String key;
        private final GGUFValueType type;
        /** The element type of an ARRAY override, null for every scalar type. */
        private final GGUFValueType elementType;
        private final Object value;

        Override(String key, GGUFValueType type, Object value) {
            this(key, type, null, value);
        }

        private Override(String key, GGUFValueType type, GGUFValueType elementType, Object value) {
            if (key == null || key.isEmpty()) {
                throw new IllegalArgumentException("an override needs a key");
            }
            if (type == GGUFValueType.ARRAY) {
                if (elementType == null || elementType == GGUFValueType.ARRAY) {
                    throw new IllegalArgumentException("override of key '" + key + "': an ARRAY override needs a"
                            + " scalar element type");
                }
                if (!(value instanceof List)) {
                    throw new IllegalArgumentException("override of key '" + key + "': an ARRAY override needs a"
                            + " list of values");
                }
                for (Object e : (List<?>) value) {
                    requireRepresentation(key, elementType, e);
                }
            } else if (elementType != null) {
                throw new IllegalArgumentException("override of key '" + key + "': only an ARRAY override takes an"
                        + " element type, " + type + " takes none");
            } else {
                requireRepresentation(key, type, value);
            }
            this.key = key;
            this.type = type;
            this.elementType = elementType;
            this.value = value;
        }

        /**
         * A scalar override of {@code type} (STRING, BOOL, INT32, UINT32, INT64, FLOAT32 or FLOAT64) whose value is
         * the type's decoded representation: String, Boolean, Integer (INT32 and UINT32), Long, Float or Double.
         *
         * @throws IllegalArgumentException naming the key for another type or a value of another representation
         */
        public static Override scalar(String key, GGUFValueType type, Object value) {
            return new Override(key, Objects.requireNonNull(type, "type"), null, value);
        }

        /**
         * An ARRAY override of {@code elementType}; {@code value} is the unmodifiable copy of the element list in
         * file order, each element in the representation {@link #scalar} takes.
         *
         * @throws IllegalArgumentException for an element type that is not scalar, or an element of another
         *                                  representation
         * @throws NullPointerException     for a null list or a null element
         */
        public static Override array(String key, GGUFValueType elementType, List<?> value) {
            Objects.requireNonNull(value, "value");
            List<Object> copy = new ArrayList<>(value.size());
            for (Object e : value) {
                copy.add(Objects.requireNonNull(e, "element"));
            }
            return new Override(key, GGUFValueType.ARRAY, elementType, Collections.unmodifiableList(copy));
        }

        /** Refuses a value that is not the decoded representation of the scalar {@code type}. */
        private static void requireRepresentation(String key, GGUFValueType type, Object value) {
            Class<?> expected;
            switch (type) {
                case STRING: expected = String.class; break;
                case BOOL: expected = Boolean.class; break;
                case UINT8:
                case INT8:
                case UINT16:
                case INT16:
                case INT32:
                case UINT32: expected = Integer.class; break;
                case INT64: expected = Long.class; break;
                case FLOAT32: expected = Float.class; break;
                case FLOAT64: expected = Double.class; break;
                default:
                    throw new IllegalArgumentException("override of key '" + key + "': " + type
                            + " is not an override type");
            }
            if (!expected.isInstance(value)) {
                throw new IllegalArgumentException("override of key '" + key + "': a " + type + " value is a "
                        + expected.getSimpleName() + ", not " + (value == null ? "null"
                        : value.getClass().getSimpleName()));
            }
        }

        /** The metadata key, written as given (it may itself carry {@code :}). */
        public String key() {
            return key;
        }

        /** The GGUF metadata type the override stores: ARRAY when it carries an element type. */
        public GGUFValueType type() {
            return type;
        }

        /** The element type of an ARRAY override, null for every scalar type. */
        public GGUFValueType elementType() {
            return elementType;
        }

        /** The value in the decoded representation of {@link #type()}; the element list for an ARRAY override. */
        public Object value() {
            return value;
        }

        @java.lang.Override
        public boolean equals(Object o) {
            if (this == o) {
                return true;
            }
            if (!(o instanceof Override)) {
                return false;
            }
            Override that = (Override) o;
            return key.equals(that.key) && type == that.type && elementType == that.elementType
                    && value.equals(that.value);
        }

        @java.lang.Override
        public int hashCode() {
            return Objects.hash(key, type, elementType, value);
        }

        /**
         * The parsed form {@code key=type:value} of a scalar override, or the informative
         * {@code key=array(type):[values]} of an ARRAY override (that form does not round-trip through
         * {@link #parse(String, String)}).
         */
        @java.lang.Override
        public String toString() {
            StringBuilder s = new StringBuilder(key).append('=');
            if (type == GGUFValueType.ARRAY) {
                s.append("array(").append(typeName(elementType)).append("):");
                return s.append(value).toString();
            }
            s.append(typeName(type)).append(':');
            return s.append(value).toString();
        }

        private String typeName(GGUFValueType t) {
            switch (t) {
                case STRING: return "str";
                case BOOL: return "bool";
                case UINT8: return "uint8";
                case INT8: return "int8";
                case UINT16: return "uint16";
                case INT16: return "int16";
                case INT32: return "int";
                case UINT32: return "uint32";
                case INT64: return "int64";
                case FLOAT32: return "float";
                case FLOAT64: return "float64";
                default: throw new IllegalStateException(t + " is not an override type");
            }
        }
    }

    /**
     * Parses one {@code key=type:value} entry.
     *
     * @param what the flag or property the text came from, named in errors
     * @throws IllegalArgumentException naming the entry when it is not of the {@code key=type:value} form, names an
     *                               unknown type, or carries a value outside its type
     */
    public static Override parse(String text, String what) {
        Objects.requireNonNull(text, what);
        String cur = text.trim();
        int eq = cur.indexOf('=');
        if (eq < 0) {
            throw bad(what, cur, "expected type:value after the key (" + TYPE_HELP + ")");
        }
        String key = cur.substring(0, eq).trim();
        if (key.isEmpty()) {
            throw bad(what, cur, "the key is empty");
        }
        String rest = cur.substring(eq + 1).trim();
        int colon = rest.indexOf(':');
        String typeName = colon < 0 ? "" : rest.substring(0, colon).trim().toLowerCase(Locale.ROOT);
        String value = colon < 0 ? "" : rest.substring(colon + 1);
        if (colon < 0) {
            throw bad(what, cur, "expected type:value after the key (" + TYPE_HELP + ")");
        }
        GGUFValueType type;
        Object parsed;
        switch (typeName) {
            case "str":
                type = GGUFValueType.STRING;
                parsed = value;
                break;
            case "bool": {
                type = GGUFValueType.BOOL;
                String b = value.trim().toLowerCase(Locale.ROOT);
                if (!b.equals("true") && !b.equals("false")) {
                    throw bad(what, cur, "bool is not true or false");
                }
                parsed = (Object) Boolean.parseBoolean(b);
                break;
            }
            case "int": {
                type = GGUFValueType.INT32;
                long i;
                try {
                    i = Long.parseLong(value.trim());
                } catch (NumberFormatException e) {
                    throw bad(what, cur, "int is not an integer");
                }
                if (i > Integer.MAX_VALUE || i < Integer.MIN_VALUE) {
                    throw bad(what, cur, "int " + i + " does not fit INT32");
                }
                parsed = (Object) (int) i;
                break;
            }
            case "float": {
                type = GGUFValueType.FLOAT32;
                float f;
                try {
                    f = Float.parseFloat(value.trim());
                } catch (NumberFormatException e) {
                    throw bad(what, cur, "float is not a number");
                }
                if (!Float.isFinite(f)) {
                    throw bad(what, cur, "float is not finite");
                }
                parsed = (Object) f;
                break;
            }
            case "uint8":
                type = GGUFValueType.UINT8;
                parsed = (Object) (int) bounded(what, cur, value, 0L, 0xFFL);
                break;
            case "int8":
                type = GGUFValueType.INT8;
                parsed = (Object) (int) bounded(what, cur, value, Byte.MIN_VALUE, Byte.MAX_VALUE);
                break;
            case "uint16":
                type = GGUFValueType.UINT16;
                parsed = (Object) (int) bounded(what, cur, value, 0L, 0xFFFFL);
                break;
            case "int16":
                type = GGUFValueType.INT16;
                parsed = (Object) (int) bounded(what, cur, value, Short.MIN_VALUE, Short.MAX_VALUE);
                break;
            case "uint32": {
                type = GGUFValueType.UINT32;
                long u = bounded(what, cur, value, 0L, 0xFFFF_FFFFL);
                parsed = (Object) (int) u;
                break;
            }
            case "int64": {
                type = GGUFValueType.INT64;
                try {
                    parsed = (Object) Long.valueOf(Long.parseLong(value.trim()));
                } catch (NumberFormatException e) {
                    throw bad(what, cur, "int64 is not an integer");
                }
                break;
            }
            case "float64": {
                type = GGUFValueType.FLOAT64;
                double d;
                try {
                    d = Double.parseDouble(value.trim());
                } catch (NumberFormatException e) {
                    throw bad(what, cur, "float64 is not a number");
                }
                if (!Double.isFinite(d)) {
                    throw bad(what, cur, "float64 is not finite");
                }
                parsed = (Object) Double.valueOf(d);
                break;
            }
            default:
                throw bad(what, cur, "the type '" + typeName + "' is unknown; use " + TYPE_HELP);
        }
        return new Override(key, type, parsed);
    }

    /** The {@code type} names {@link #parse} accepts, for its error messages. */
    private static final String TYPE_HELP =
            "str, bool, int, float, uint8, int8, uint16, int16, uint32, int64 or float64";

    /** A decimal integer literal in {@code [lo, hi]}. */
    private static long bounded(String what, String entry, String text, long lo, long hi) {
        long v;
        try {
            v = Long.parseLong(text.trim());
        } catch (NumberFormatException e) {
            throw bad(what, entry, "the value is not an integer in [" + lo + ", " + hi + "]");
        }
        if (v < lo || v > hi) {
            throw bad(what, entry, "the value " + v + " is outside [" + lo + ", " + hi + "]");
        }
        return v;
    }

    /**
     * Every entry of {@code text}: the same list form as the {@code --override-kv} flag in order. The separator
     * between entries is {@code ;} (values keep both {@code ,} and {@code =}).
     *
     * @param what the flag or property the text came from, named in errors
     * @return the parsed overrides in order; never empty
     * @throws IllegalArgumentException naming the entry when one of them cannot be parsed, or naming {@code what}
     *                               when the list is empty
     */
    public static List<Override> parseList(String text, String what) {
        if (text == null || text.trim().isEmpty()) {
            throw new IllegalArgumentException(what + " is empty; it needs at least one key=type:value entry");
        }
        String entries = text.trim();
        List<Override> out = new ArrayList<>();
        int begin = 0;
        while (true) {
            int end = entries.indexOf(';', begin);
            String entry = (end < 0 ? entries.substring(begin) : entries.substring(begin, end)).trim();
            if (entry.isEmpty()) {
                throw new IllegalArgumentException(what + " holds an empty entry between ';'");
            }
            out.add(parse(entry, what));
            if (end < 0) {
                break;
            }
            begin = end + 1;
        }
        return List.copyOf(out);
    }

    /** The unmodifiable, single-property list form of {@code overrides}, for errors and properties. */
    public static String toString(List<Override> overrides) {
        StringBuilder s = new StringBuilder();
        for (Override o : overrides) {
            if (s.length() > 0) {
                s.append(';');
            }
            s.append(o);
        }
        return s.toString();
    }

    private static IllegalArgumentException bad(String what, String entry, String reason) {
        return new IllegalArgumentException(what + ": '" + entry + "' is not a valid key=type:value entry: " + reason);
    }
}

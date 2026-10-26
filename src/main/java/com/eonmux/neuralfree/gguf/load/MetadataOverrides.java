package com.eonmux.neuralfree.gguf.load;

import com.eonmux.neuralfree.gguf.GGUFValueType;

import java.util.ArrayList;
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
 * Parsers reject an entry without {@code type:value}, an unknown type, or a value of the wrong form, naming the
 * entry (and, in a CLI such as {@code LLMDemoOptions} with its own {@code --override-kv} flag, the flag).
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
        private final Object value;

        Override(String key, GGUFValueType type, Object value) {
            this.key = key;
            this.type = type;
            this.value = value;
        }

        /** The metadata key, written as given (it may itself carry {@code :}). */
        public String key() {
            return key;
        }

        /** The GGUF metadata type the override stores. */
        public GGUFValueType type() {
            return type;
        }

        /** The value in the decoded representation of {@link #type()}. */
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
            return key.equals(that.key) && type == that.type && value.equals(that.value);
        }

        @java.lang.Override
        public int hashCode() {
            return Objects.hash(key, type, value);
        }

        /** The parsed form: {@code key=type:value}. */
        @java.lang.Override
        public String toString() {
            StringBuilder s = new StringBuilder(key).append('=').append(typeName()).append(':');
            return s.append(value).toString();
        }

        private String typeName() {
            switch (type) {
                case STRING: return "str";
                case BOOL: return "bool";
                case INT32: return "int";
                case FLOAT32: return "float";
                default: throw new IllegalStateException(type + " is not an override type");
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
        if (eq <= 0) {
            throw bad(what, cur, "there is no '=' between the key and the type");
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
            throw bad(what, cur, "expected type:value after the key (str, bool, int or float)");
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
            default:
                throw bad(what, cur, "the type '" + typeName + "' is unknown; use str, bool, int or float");
        }
        return new Override(key, type, parsed);
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
            String entry = (end < 0 ? entries : entries.substring(begin, end)).trim();
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

package com.eonmux.neuralfree.utils.json;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.MathContext;
import java.math.RoundingMode;
import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * JSON reader and writer compatible with Python's {@code json} module.
 *
 * <p>Parsing produces the value model {@link java.util.LinkedHashMap} (String keys, document order), {@link ArrayList},
 * {@link Long} (integers that fit 64 bits), {@link BigInteger} (longer integers), {@link Double} (literals with a
 * fraction or exponent, and Python's {@code NaN}, {@code Infinity} and {@code -Infinity}), {@link String},
 * {@link Boolean} and {@code null}. The grammar is RFC 8259 plus those three constants; every document is bounded
 * by {@link Limits}, and a violation throws {@link JsonException} with its position.</p>
 *
 * <p>Writing follows {@code json.dumps}: {@link Style} carries its options and floats print as Python
 * {@code repr(float)} ({@link #floatRepr(double)}).</p>
 */
public final class Json {

    /** Longest integer literal accepted, Python's {@code sys.int_info.default_max_str_digits}. */
    public static final int MAX_INTEGER_DIGITS = 4300;

    /** Largest {@code maxBytes} accepted by {@link #parse(Path, long, Limits)} (the largest Java array). */
    public static final long MAX_FILE_BYTES = Integer.MAX_VALUE - 8;

    private static final char BOM = '\uFEFF';
    private static final char[] HEX = "0123456789abcdef".toCharArray();

    private Json() {
    }

    // ================================================================= limits

    /** What happens when an object repeats a key. */
    public enum DuplicateKeys {
        /** The document is rejected. */
        REJECT,
        /** The last value wins at the first key's position, and each duplicate is logged at DEBUG with its path. */
        LAST_WINS_LOGGED,
        /** The last value wins at the first key's position (Python {@code json.loads}). */
        LAST_WINS
    }

    /** Immutable bounds applied while parsing one document. */
    public static final class Limits {

        public static final int DEFAULT_MAX_DEPTH = 512;
        public static final long DEFAULT_MAX_VALUES = 50_000_000L;
        public static final int DEFAULT_MAX_STRING_CHARS = 268_435_456;

        /** Default caps; duplicate keys are rejected. */
        public static final Limits STRICT = new Limits(DEFAULT_MAX_DEPTH, DEFAULT_MAX_VALUES,
                DEFAULT_MAX_STRING_CHARS, DuplicateKeys.REJECT);
        /** Default caps; the last duplicate wins and each duplicate is logged. */
        public static final Limits LENIENT = STRICT.withDuplicateKeys(DuplicateKeys.LAST_WINS_LOGGED);
        /** Default caps; the last duplicate wins silently, as Python {@code json.loads}. */
        public static final Limits PYTHON = STRICT.withDuplicateKeys(DuplicateKeys.LAST_WINS);

        private final int maxDepth;
        private final long maxValues;
        private final int maxStringChars;
        private final DuplicateKeys duplicateKeys;

        /**
         * @param maxDepth       deepest nesting of objects and arrays (the outermost container is depth 1)
         * @param maxValues      most values in a document, containers included and object keys excluded
         * @param maxStringChars longest string or key, in UTF-16 chars after unescaping
         * @param duplicateKeys  policy for repeated object keys
         */
        public Limits(int maxDepth, long maxValues, int maxStringChars, DuplicateKeys duplicateKeys) {
            if (maxDepth < 1) throw new IllegalArgumentException("maxDepth must be >= 1, got " + maxDepth);
            if (maxValues < 1) throw new IllegalArgumentException("maxValues must be >= 1, got " + maxValues);
            if (maxStringChars < 0) throw new IllegalArgumentException("maxStringChars must be >= 0, got " + maxStringChars);
            this.maxDepth = maxDepth;
            this.maxValues = maxValues;
            this.maxStringChars = maxStringChars;
            this.duplicateKeys = Objects.requireNonNull(duplicateKeys, "duplicateKeys");
        }

        public int maxDepth() { return maxDepth; }

        public long maxValues() { return maxValues; }

        public int maxStringChars() { return maxStringChars; }

        public DuplicateKeys duplicateKeys() { return duplicateKeys; }

        public Limits withMaxDepth(int value) { return new Limits(value, maxValues, maxStringChars, duplicateKeys); }

        public Limits withMaxValues(long value) { return new Limits(maxDepth, value, maxStringChars, duplicateKeys); }

        public Limits withMaxStringChars(int value) { return new Limits(maxDepth, maxValues, value, duplicateKeys); }

        public Limits withDuplicateKeys(DuplicateKeys value) { return new Limits(maxDepth, maxValues, maxStringChars, value); }

        @Override
        public String toString() {
            return "Limits{maxDepth=" + maxDepth + ", maxValues=" + maxValues + ", maxStringChars=" + maxStringChars
                    + ", duplicateKeys=" + duplicateKeys + '}';
        }
    }

    // ================================================================= parsing

    /** Parses {@code text} under {@link Limits#STRICT}. */
    public static Object parse(String text) {
        return parse(text, Limits.STRICT);
    }

    /** Parses one JSON document; whitespace may surround it, anything else after it is an error. */
    public static Object parse(String text, Limits limits) {
        Objects.requireNonNull(text, "text");
        return new JsonParser(text, 0, Objects.requireNonNull(limits, "limits"), null).document();
    }

    /**
     * Parses the JSON document in {@code file}: strict UTF-8 (malformed bytes are an error, not replaced), with an
     * optional leading byte order mark, at most {@code maxBytes} long. Errors carry the file name in their message;
     * positions are byte offsets for encoding errors and character offsets into the decoded text otherwise.
     *
     * @throws IOException   when the file cannot be read
     * @throws JsonException when the file is too large, is not UTF-8 or is not a valid document within {@code limits}
     */
    public static Object parse(Path file, long maxBytes, Limits limits) throws IOException {
        Objects.requireNonNull(file, "file");
        Objects.requireNonNull(limits, "limits");
        if (maxBytes < 0 || maxBytes > MAX_FILE_BYTES) {
            throw new IllegalArgumentException("maxBytes must be in [0, " + MAX_FILE_BYTES + "], got " + maxBytes);
        }
        String source = file.toString();
        String text = decodeUtf8(readBounded(file, maxBytes, source), source);
        int start = !text.isEmpty() && text.charAt(0) == BOM ? 1 : 0;
        try {
            return new JsonParser(text, start, limits, source).document();
        } catch (JsonException e) {
            throw e.withSource(source);
        }
    }

    private static byte[] readBounded(Path file, long maxBytes, String source) throws IOException {
        long size = Files.size(file);
        if (size > maxBytes) {
            throw new JsonException(source + ": size " + size + " bytes exceeds the limit of " + maxBytes + " bytes");
        }
        try (InputStream in = Files.newInputStream(file)) {
            byte[] bytes = in.readNBytes((int) maxBytes);
            if (in.read() != -1) {
                throw new JsonException(source + ": size exceeds the limit of " + maxBytes + " bytes");
            }
            return bytes;
        }
    }

    private static String decodeUtf8(byte[] bytes, String source) {
        CharsetDecoder decoder = StandardCharsets.UTF_8.newDecoder()
                .onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT);
        ByteBuffer in = ByteBuffer.wrap(bytes);
        try {
            return decoder.decode(in).toString();
        } catch (CharacterCodingException e) {
            throw new JsonException(source + ": invalid UTF-8 byte sequence", in.position(), e);
        }
    }

    // ================================================================= writing

    /** Immutable {@code json.dumps} options. */
    public static final class Style {

        /** {@code json.dumps} defaults: {@code ", "}, {@code ": "}, ASCII-only output, no indent, insertion order. */
        public static final Style PYTHON_DEFAULT = new Style(true, null, ", ", ": ", false);
        /** Compact, key-sorted, non-ASCII kept: one text per value, for comparison and hashing. */
        public static final Style CANONICAL = new Style(false, null, ",", ":", true);

        private final boolean ensureAscii;
        private final String indent;
        private final String itemSep;
        private final String keySep;
        private final boolean sortKeys;

        /**
         * @param ensureAscii escape every character outside {@code ' '..'~'}
         * @param indent      text repeated per nesting level after each newline, or null for single-line output
         * @param itemSep     separator between items and members
         * @param keySep      separator between a key and its value
         * @param sortKeys    write object members ordered by key
         */
        public Style(boolean ensureAscii, String indent, String itemSep, String keySep, boolean sortKeys) {
            this.ensureAscii = ensureAscii;
            this.indent = indent;
            this.itemSep = Objects.requireNonNull(itemSep, "itemSep");
            this.keySep = Objects.requireNonNull(keySep, "keySep");
            this.sortKeys = sortKeys;
        }

        public boolean ensureAscii() { return ensureAscii; }

        /** Null for single-line output. */
        public String indent() { return indent; }

        public String itemSep() { return itemSep; }

        public String keySep() { return keySep; }

        public boolean sortKeys() { return sortKeys; }
    }

    /**
     * Maps a host value model onto the written one. The writer passes every value through {@link #value(Object)}
     * and then accepts null, {@link Boolean}, {@link String}, {@link Long}, {@link Integer}, {@link Short},
     * {@link Byte}, {@link BigInteger}, {@link Double}, {@link Float}, {@link List} and {@link Map}.
     */
    public interface Adapter {

        /** The identity mapping with Python key coercion and ordering. */
        Adapter DEFAULT = new Adapter() {
        };

        /** The value to write in place of {@code v}. */
        default Object value(Object v) {
            return v;
        }

        /** The text of map key {@code k}; see {@link Json#keyText(Object)}. */
        default String key(Object k) {
            return keyText(k);
        }

        /** Order of two map keys for {@link Style#sortKeys()}; see {@link Json#compareKeys(Object, Object)}. */
        default int compareKeys(Object a, Object b) {
            return Json.compareKeys(a, b);
        }
    }

    /** Writes {@code value} to {@code out}. */
    public static void write(Object value, Style style, Appendable out) throws IOException {
        write(value, style, Adapter.DEFAULT, out);
    }

    /** Writes {@code value}, mapped through {@code adapter}, to {@code out}. */
    public static void write(Object value, Style style, Adapter adapter, Appendable out) throws IOException {
        new Writer(Objects.requireNonNull(style, "style"), Objects.requireNonNull(adapter, "adapter"),
                Objects.requireNonNull(out, "out")).value(value, 0);
    }

    /** The JSON text of {@code value}. */
    public static String toString(Object value, Style style) {
        return toString(value, style, Adapter.DEFAULT);
    }

    /** The JSON text of {@code value}, mapped through {@code adapter}. */
    public static String toString(Object value, Style style, Adapter adapter) {
        StringBuilder sb = new StringBuilder();
        try {
            write(value, style, adapter, sb);
        } catch (IOException e) {
            throw new UncheckedIOException("StringBuilder append failed", e);
        }
        return sb.toString();
    }

    /**
     * Python {@code json} key coercion: strings as-is; {@code None}, booleans and numbers by their JSON text.
     *
     * @throws JsonException for any other key type
     */
    public static String keyText(Object k) {
        if (k instanceof String) return (String) k;
        if (k == null) return "null";
        if (k instanceof Boolean) return (Boolean) k ? "true" : "false";
        if (isIntegral(k)) return k.toString();
        if (k instanceof Double || k instanceof Float) return numberText(((Number) k).doubleValue());
        throw new JsonException("keys must be str, int, float, bool or None, not " + k.getClass().getSimpleName());
    }

    /**
     * Python {@code sorted()} order of map keys: strings by code point, numbers (booleans as 0 and 1) by value.
     *
     * @throws JsonException for keys that Python cannot order against each other
     */
    public static int compareKeys(Object a, Object b) {
        if (a instanceof String && b instanceof String) return compareCodePoints((String) a, (String) b);
        if (isNumberKey(a) && isNumberKey(b)) {
            double da = numberKeyAsDouble(a), db = numberKeyAsDouble(b);
            if (Double.isNaN(da) || Double.isNaN(db) || Double.isInfinite(da) || Double.isInfinite(db)) {
                return Double.compare(da, db);
            }
            return numberKeyAsDecimal(a).compareTo(numberKeyAsDecimal(b));
        }
        throw new JsonException("cannot order keys of types " + typeOf(a) + " and " + typeOf(b));
    }

    private static boolean isIntegral(Object v) {
        return v instanceof Long || v instanceof Integer || v instanceof Short || v instanceof Byte
                || v instanceof BigInteger;
    }

    private static boolean isNumberKey(Object v) {
        return v instanceof Boolean || isIntegral(v) || v instanceof Double || v instanceof Float;
    }

    private static double numberKeyAsDouble(Object v) {
        return v instanceof Boolean ? ((Boolean) v ? 1.0 : 0.0) : ((Number) v).doubleValue();
    }

    private static BigDecimal numberKeyAsDecimal(Object v) {
        if (v instanceof Boolean) return (Boolean) v ? BigDecimal.ONE : BigDecimal.ZERO;
        if (v instanceof BigInteger) return new BigDecimal((BigInteger) v);
        if (v instanceof Double || v instanceof Float) return new BigDecimal(((Number) v).doubleValue());
        return BigDecimal.valueOf(((Number) v).longValue());
    }

    private static String typeOf(Object v) {
        return v == null ? "None" : v.getClass().getSimpleName();
    }

    private static int compareCodePoints(String a, String b) {
        int i = 0, j = 0;
        while (i < a.length() && j < b.length()) {
            int ca = a.codePointAt(i), cb = b.codePointAt(j);
            if (ca != cb) return Integer.compare(ca, cb);
            i += Character.charCount(ca);
            j += Character.charCount(cb);
        }
        return Integer.compare(a.length() - i, b.length() - j);
    }

    /** JSON text of a float: Python's {@code NaN}, {@code Infinity}, {@code -Infinity}, else {@link #floatRepr}. */
    public static String numberText(double d) {
        if (Double.isNaN(d)) return "NaN";
        if (Double.isInfinite(d)) return d > 0 ? "Infinity" : "-Infinity";
        return floatRepr(d);
    }

    /**
     * Python {@code repr(float)}: the shortest decimal that rounds back to {@code d} (the closest one when several
     * have that length), in fixed notation for decimal exponents in [-4, 16) and as {@code d.ddde±XX} otherwise;
     * {@code nan}, {@code inf}, {@code -inf}, {@code 0.0} and {@code -0.0} for the special values.
     */
    public static String floatRepr(double d) {
        if (Double.isNaN(d)) return "nan";
        if (Double.isInfinite(d)) return d > 0 ? "inf" : "-inf";
        if (d == 0.0) return (1.0 / d) < 0 ? "-0.0" : "0.0";
        BigDecimal bd = shortestDecimal(Math.abs(d));
        String digits = bd.unscaledValue().toString();
        int n = digits.length();
        int exp10 = n - 1 - bd.scale();
        StringBuilder out = new StringBuilder(d < 0 ? "-" : "");
        if (exp10 >= -4 && exp10 < 16) {
            int point = exp10 + 1;
            if (point <= 0) {
                out.append("0.");
                for (int i = 0; i < -point; i++) out.append('0');
                out.append(digits);
            } else if (point >= n) {
                out.append(digits);
                for (int i = 0; i < point - n; i++) out.append('0');
                out.append(".0");
            } else {
                out.append(digits, 0, point).append('.').append(digits, point, n);
            }
        } else {
            out.append(digits.charAt(0));
            if (n > 1) out.append('.').append(digits, 1, n);
            out.append('e').append(exp10 < 0 ? '-' : '+');
            int ae = Math.abs(exp10);
            if (ae < 10) out.append('0');
            out.append(ae);
        }
        return out.toString();
    }

    /**
     * The shortest round-tripping decimal of finite {@code a > 0}, as David Gay's {@code dtoa} mode 0 behind Python's
     * {@code repr}: for the fewest significant digits {@code p} at which one of the two {@code p}-digit neighbours of
     * the exact binary value parses back to {@code a}, that neighbour, or the nearer one (ties to even) when both do.
     * Both neighbours are tried because the rounding interval of a power of two is narrower below it than above, so
     * the nearer neighbour can fall outside it while the farther one is inside. Seventeen digits always round-trip.
     */
    private static BigDecimal shortestDecimal(double a) {
        BigDecimal exact = new BigDecimal(a);
        for (int p = 1; ; p++) {
            BigDecimal down = exact.round(new MathContext(p, RoundingMode.DOWN));
            BigDecimal up = exact.round(new MathContext(p, RoundingMode.UP));
            boolean downOk = down.doubleValue() == a;
            boolean upOk = up.doubleValue() == a;
            if (downOk && upOk) return exact.round(new MathContext(p, RoundingMode.HALF_EVEN)).stripTrailingZeros();
            if (downOk) return down.stripTrailingZeros();
            if (upOk) return up.stripTrailingZeros();
        }
    }

    /** One {@code json.dumps} pass; detects reference cycles as Python's {@code check_circular} does. */
    private static final class Writer {
        private final Style st;
        private final Adapter adapter;
        private final Appendable out;
        private final Set<Object> active = Collections.newSetFromMap(new IdentityHashMap<>());

        Writer(Style st, Adapter adapter, Appendable out) {
            this.st = st;
            this.adapter = adapter;
            this.out = out;
        }

        void value(Object raw, int depth) throws IOException {
            Object v = adapter.value(raw);
            if (v == null) out.append("null");
            else if (v instanceof Boolean) out.append((Boolean) v ? "true" : "false");
            else if (v instanceof String) string((String) v);
            else if (isIntegral(v)) out.append(v.toString());
            else if (v instanceof Double || v instanceof Float) out.append(numberText(((Number) v).doubleValue()));
            else if (v instanceof List) list((List<?>) v, depth);
            else if (v instanceof Map) map((Map<?, ?>) v, depth);
            else throw new JsonException("Object of type " + v.getClass().getSimpleName() + " is not JSON serializable");
        }

        private void enter(Object container) {
            if (!active.add(container)) throw new JsonException("Circular reference detected");
        }

        private void newline(int depth) throws IOException {
            if (st.indent == null) return;
            out.append('\n');
            for (int i = 0; i < depth; i++) out.append(st.indent);
        }

        private void list(List<?> list, int depth) throws IOException {
            if (list.isEmpty()) {
                out.append("[]");
                return;
            }
            enter(list);
            out.append('[');
            int i = 0;
            for (Object item : list) {
                if (i++ > 0) out.append(st.itemSep);
                newline(depth + 1);
                value(item, depth + 1);
            }
            newline(depth);
            out.append(']');
            active.remove(list);
        }

        private void map(Map<?, ?> map, int depth) throws IOException {
            if (map.isEmpty()) {
                out.append("{}");
                return;
            }
            enter(map);
            List<Map.Entry<?, ?>> entries = new ArrayList<>(map.entrySet());
            if (st.sortKeys) entries.sort((a, b) -> adapter.compareKeys(a.getKey(), b.getKey()));
            out.append('{');
            boolean first = true;
            for (Map.Entry<?, ?> e : entries) {
                if (!first) out.append(st.itemSep);
                first = false;
                newline(depth + 1);
                string(adapter.key(e.getKey()));
                out.append(st.keySep);
                value(e.getValue(), depth + 1);
            }
            newline(depth);
            out.append('}');
            active.remove(map);
        }

        private void string(String s) throws IOException {
            out.append('"');
            int run = 0;
            for (int i = 0; i < s.length(); i++) {
                char c = s.charAt(i);
                if (c >= 0x20 && c != '"' && c != '\\' && (!st.ensureAscii || c <= 0x7e)) continue;
                out.append(s, run, i);
                run = i + 1;
                switch (c) {
                    case '"': out.append("\\\""); break;
                    case '\\': out.append("\\\\"); break;
                    case '\n': out.append("\\n"); break;
                    case '\r': out.append("\\r"); break;
                    case '\t': out.append("\\t"); break;
                    case '\b': out.append("\\b"); break;
                    case '\f': out.append("\\f"); break;
                    default:
                        out.append("\\u").append(HEX[(c >> 12) & 0xF]).append(HEX[(c >> 8) & 0xF])
                                .append(HEX[(c >> 4) & 0xF]).append(HEX[c & 0xF]);
                }
            }
            out.append(s, run, s.length()).append('"');
        }
    }
}

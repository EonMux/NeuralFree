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

    /**
     * Immutable bounds applied while parsing one document. The seven fields cover an untrusted document end to
     * end: every check precedes the allocation or recursion it bounds.
     */
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

        /**
         * Caps for input from outside the trust boundary (§3.6): depth 128, 16 MiB, 4 Mi values, 8 Mi UTF-16 chars
         * per string, 1 Mi members per container; duplicate keys are rejected and NaN, Infinity, -Infinity too.
         */
        public static final Limits UNTRUSTED = new Limits(128, 4_194_304L, 8_388_608, DuplicateKeys.REJECT,
                16_777_216L, 1_048_576, false);

        private final int maxDepth;
        private final long maxValues;
        private final int maxStringChars;
        private final long maxBytes;
        private final int maxMembers;
        private final DuplicateKeys duplicateKeys;
        private final boolean allowNonFinite;

        /**
         * Bounds with the bytes, members and constants fields unbounded, so the constants built above and every
         * existing caller keep HF.md's behaviour.
         *
         * @param maxDepth       deepest nesting of objects and arrays (the outermost container is depth 1)
         * @param maxValues      most values in a document, containers included and object keys excluded
         * @param maxStringChars longest string or key, in UTF-16 chars after unescaping
         * @param duplicateKeys  policy for repeated object keys
         */
        public Limits(int maxDepth, long maxValues, int maxStringChars, DuplicateKeys duplicateKeys) {
            this(maxDepth, maxValues, maxStringChars, duplicateKeys, Long.MAX_VALUE, Integer.MAX_VALUE, true);
        }

        /**
         * @param maxDepth       deepest nesting of objects and arrays (the outermost container is depth 1)
         * @param maxValues      most values in a document, containers included and object keys excluded
         * @param maxStringChars longest string or key, in UTF-16 chars after unescaping
         * @param duplicateKeys  policy for repeated object keys
         * @param maxBytes       longest input, in bytes for byte and file input and in UTF-8 bytes for a CharSequence
         * @param maxMembers     most members of one object or array
         * @param allowNonFinite whether the {@code NaN}, {@code Infinity} and {@code -Infinity} literals parse
         */
        public Limits(int maxDepth, long maxValues, int maxStringChars, DuplicateKeys duplicateKeys,
                long maxBytes, int maxMembers, boolean allowNonFinite) {
            if (maxDepth < 1) throw new IllegalArgumentException("maxDepth must be >= 1, got " + maxDepth);
            if (maxValues < 1) throw new IllegalArgumentException("maxValues must be >= 1, got " + maxValues);
            if (maxStringChars < 0) throw new IllegalArgumentException("maxStringChars must be >= 0, got " + maxStringChars);
            if (maxBytes < 0) throw new IllegalArgumentException("maxBytes must be >= 0, got " + maxBytes);
            if (maxMembers < 0) throw new IllegalArgumentException("maxMembers must be >= 0, got " + maxMembers);
            this.maxDepth = maxDepth;
            this.maxValues = maxValues;
            this.maxStringChars = maxStringChars;
            this.maxBytes = maxBytes;
            this.maxMembers = maxMembers;
            this.duplicateKeys = Objects.requireNonNull(duplicateKeys, "duplicateKeys");
            this.allowNonFinite = allowNonFinite;
        }

        public int maxDepth() { return maxDepth; }

        public long maxValues() { return maxValues; }

        public int maxStringChars() { return maxStringChars; }

        /** Longest input: bytes for byte and file input, UTF-8 bytes for a CharSequence. */
        public long maxBytes() { return maxBytes; }

        /** Most members of one object or array. */
        public int maxMembers() { return maxMembers; }

        public DuplicateKeys duplicateKeys() { return duplicateKeys; }

        /** Whether the {@code NaN}, {@code Infinity} and {@code -Infinity} literals parse. */
        public boolean allowNonFinite() { return allowNonFinite; }

        public Limits withMaxDepth(int value) {
            return new Limits(value, maxValues, maxStringChars, duplicateKeys, maxBytes, maxMembers, allowNonFinite);
        }

        public Limits withMaxValues(long value) {
            return new Limits(maxDepth, value, maxStringChars, duplicateKeys, maxBytes, maxMembers, allowNonFinite);
        }

        public Limits withMaxStringChars(int value) {
            return new Limits(maxDepth, maxValues, value, duplicateKeys, maxBytes, maxMembers, allowNonFinite);
        }

        public Limits withMaxBytes(long value) {
            return new Limits(maxDepth, maxValues, maxStringChars, duplicateKeys, value, maxMembers, allowNonFinite);
        }

        public Limits withMaxMembers(int value) {
            return new Limits(maxDepth, maxValues, maxStringChars, duplicateKeys, maxBytes, value, allowNonFinite);
        }

        public Limits withDuplicateKeys(DuplicateKeys value) {
            return new Limits(maxDepth, maxValues, maxStringChars, value, maxBytes, maxMembers, allowNonFinite);
        }

        public Limits withAllowNonFinite(boolean value) {
            return new Limits(maxDepth, maxValues, maxStringChars, duplicateKeys, maxBytes, maxMembers, value);
        }

        @Override
        public String toString() {
            return "Limits{maxDepth=" + maxDepth + ", maxValues=" + maxValues + ", maxStringChars=" + maxStringChars
                    + ", maxBytes=" + maxBytes + ", maxMembers=" + maxMembers + ", duplicateKeys=" + duplicateKeys
                    + ", allowNonFinite=" + allowNonFinite + '}';
        }
    }

    // ================================================================= parsing

    /** Parses {@code text} under {@link Limits#STRICT}. */
    public static Object parse(String text) {
        return parse(text, Limits.STRICT);
    }

    /** Parses one JSON document; whitespace may surround it, anything else after it is an error. */
    public static Object parse(String text, Limits limits) {
        return parse((CharSequence) text, limits);
    }

    /**
     * Parses one JSON document over any {@link CharSequence}; whitespace may surround it, anything else after it is
     * an error. {@code limits.maxBytes()} bounds the UTF-8 byte length of the text, which is therefore checked before
     * a non-{@link String} text is copied into one.
     */
    public static Object parse(CharSequence text, Limits limits) {
        Objects.requireNonNull(text, "text");
        Objects.requireNonNull(limits, "limits");
        long bytes = utf8Length(text);
        if (bytes > limits.maxBytes()) {
            throw new JsonException("input length " + bytes + " bytes exceeds maxBytes " + limits.maxBytes());
        }
        return new JsonParser(text instanceof String ? (String) text : text.toString(), 0, limits, null).document();
    }

    /** UTF-8 byte length of {@code text}, without allocating; lone surrogates count as their escaped length. */
    private static long utf8Length(CharSequence text) {
        long bytes = 0;
        for (int i = 0, n = text.length(); i < n; ) {
            char c = text.charAt(i++);
            if (c < 0x80) bytes++;
            else if (c < 0x800) bytes += 2;
            else if (Character.isHighSurrogate(c) && i < n && Character.isLowSurrogate(text.charAt(i))) {
                i++;
                bytes += 4;
            } else bytes += 3;
        }
        return bytes;
    }

    /**
     * Parses the JSON document in {@code bytes[off, off+len)}: strict UTF-8 (malformed bytes are an error, not
     * replaced), with an optional leading byte order mark, at most {@code limits.maxBytes()} long — the length check
     * precedes the decode. Positions are byte offsets for encoding errors and character offsets into the decoded
     * text otherwise.
     *
     * @throws JsonException when the range is too large, is not UTF-8 or is not a valid document within {@code limits}
     */
    public static Object parse(byte[] bytes, int off, int len, Limits limits) {
        Objects.requireNonNull(bytes, "bytes");
        Objects.requireNonNull(limits, "limits");
        if (off < 0 || len < 0 || off > bytes.length - len) {
            throw new IllegalArgumentException(
                    "off=" + off + ", len=" + len + " is not a range inside [0, " + bytes.length + ']');
        }
        if (len > limits.maxBytes()) {
            throw new JsonException("input length " + len + " bytes exceeds maxBytes " + limits.maxBytes());
        }
        String text = decodeUtf8(bytes, off, len, null);
        int start = !text.isEmpty() && text.charAt(0) == BOM ? 1 : 0;
        return new JsonParser(text, start, limits, null).document();
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
        byte[] bytes = readBounded(file, maxBytes, source);
        String text = decodeUtf8(bytes, 0, bytes.length, source);
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

    private static String decodeUtf8(byte[] bytes, int off, int len, String source) {
        CharsetDecoder decoder = StandardCharsets.UTF_8.newDecoder()
                .onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT);
        ByteBuffer in = ByteBuffer.wrap(bytes, off, len);
        try {
            return decoder.decode(in).toString();
        } catch (CharacterCodingException e) {
            throw new JsonException((source == null ? "" : source + ": ") + "invalid UTF-8 byte sequence",
                    in.position(), e);
        }
    }

    // ================================================================= writing

    /** Immutable {@code json.dumps} options. */
    public static final class Style {

        /** {@code json.dumps} defaults: {@code ", "}, {@code ": "}, ASCII-only output, no indent, insertion order. */
        public static final Style PYTHON_DEFAULT = new Style(true, null, ", ", ": ", false);
        /** Compact, key-sorted, non-ASCII kept: one text per value, for comparison and hashing. */
        public static final Style CANONICAL = new Style(false, null, ",", ":", true);
        /** Serving style: compact ({@code ","}, {@code ":"}), non-ASCII kept, insertion order, U+2028 and U+2029
         *  escaped so a value stays on one SSE {@code data:} line, non-finite doubles rejected. */
        public static final Style API = new Style(false, null, ",", ":", false, true, true);

        private final boolean ensureAscii;
        private final String indent;
        private final String itemSep;
        private final String keySep;
        private final boolean sortKeys;
        private final boolean escapeLineSeparators;
        private final boolean rejectNonFinite;

        /** The options above with the two serving flags false, exactly as before they existed. */
        public Style(boolean ensureAscii, String indent, String itemSep, String keySep, boolean sortKeys) {
            this(ensureAscii, indent, itemSep, keySep, sortKeys, false, false);
        }

        /**
         * @param ensureAscii          escape every character outside {@code ' '..'~'}
         * @param indent               text repeated per nesting level after each newline, or null for single-line output
         * @param itemSep              separator between items and members
         * @param keySep               separator between a key and its value
         * @param sortKeys             write object members ordered by key
         * @param escapeLineSeparators write U+2028 and U+2029 as {@code \u2028}/{@code \u2029} even when
         *                             {@code ensureAscii} is false, so a value stays on one line
         * @param rejectNonFinite      throw {@link IllegalArgumentException} for a non-finite double; callers clamp
         */
        public Style(boolean ensureAscii, String indent, String itemSep, String keySep, boolean sortKeys,
                boolean escapeLineSeparators, boolean rejectNonFinite) {
            this.ensureAscii = ensureAscii;
            this.indent = indent;
            this.itemSep = Objects.requireNonNull(itemSep, "itemSep");
            this.keySep = Objects.requireNonNull(keySep, "keySep");
            this.sortKeys = sortKeys;
            this.escapeLineSeparators = escapeLineSeparators;
            this.rejectNonFinite = rejectNonFinite;
        }

        public boolean ensureAscii() { return ensureAscii; }

        /** Null for single-line output. */
        public String indent() { return indent; }

        public String itemSep() { return itemSep; }

        public String keySep() { return keySep; }

        public boolean sortKeys() { return sortKeys; }

        public boolean escapeLineSeparators() { return escapeLineSeparators; }

        public boolean rejectNonFinite() { return rejectNonFinite; }
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

    /**
     * Writes the JSON text of the double {@code d} to {@code out}, formatted as {@link #numberText(double)}. The
     * shared float path of {@link Json#write} and {@link JsonStreamWriter}; {@link Style#rejectNonFinite()} turns a
     * non-finite {@code d} into an {@link IllegalArgumentException} instead of invalid JSON.
     */
    static void writeNumber(double d, Style st, Appendable out) throws IOException {
        if (st.rejectNonFinite && !Double.isFinite(d)) {
            throw new IllegalArgumentException("non-finite double is not valid JSON: " + numberText(d));
        }
        out.append(numberText(d));
    }

    /** Writes {@code s} as a JSON string, applying {@link Style}'s escaping; the shared string path above and below. */
    static void writeString(String s, Style st, Appendable out) throws IOException {
        out.append('"');
        int run = 0;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            boolean escape = c < 0x20 || c == '"' || c == '\\' || (st.ensureAscii && c > 0x7e)
                    || (st.escapeLineSeparators && (c == '\u2028' || c == '\u2029'));
            if (!escape) continue;
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
                case '\u2028': out.append("\\u2028"); break;
                case '\u2029': out.append("\\u2029"); break;
                default:
                    out.append("\\u").append(HEX[(c >> 12) & 0xF]).append(HEX[(c >> 8) & 0xF])
                            .append(HEX[(c >> 4) & 0xF]).append(HEX[c & 0xF]);
            }
        }
        out.append(s, run, s.length()).append('"');
    }

    /** One indent newline as {@link Json#write} writes it: nothing without an indent, else a newline and {@code depth} indents. */
    static void newline(Appendable out, String indent, int depth) throws IOException {
        if (indent == null) return;
        out.append('\n');
        for (int i = 0; i < depth; i++) out.append(indent);
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
            else if (v instanceof String) writeString((String) v, st, out);
            else if (isIntegral(v)) out.append(v.toString());
            else if (v instanceof Double || v instanceof Float) writeNumber(((Number) v).doubleValue(), st, out);
            else if (v instanceof List) list((List<?>) v, depth);
            else if (v instanceof Map) map((Map<?, ?>) v, depth);
            else throw new JsonException("Object of type " + v.getClass().getSimpleName() + " is not JSON serializable");
        }

        private void enter(Object container) {
            if (!active.add(container)) throw new JsonException("Circular reference detected");
        }

        private void newline(int depth) throws IOException {
            Json.newline(out, st.indent(), depth);
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
                writeString(adapter.key(e.getKey()), st, out);
                out.append(st.keySep);
                value(e.getValue(), depth + 1);
            }
            newline(depth);
            out.append('}');
            active.remove(map);
        }
    }
}

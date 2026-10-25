package com.eonmux.neuralfree.gguf.tokenizer;

import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * An immutable Hugging Face normalizer chain (tokenizers {@code normalizers}): Unicode
 * normalization forms, Lowercase, StripAccents, Strip, Replace, Prepend, Precompiled,
 * BertNormalizer, Nmt, ByteLevel and Sequence, with Hugging Face's character semantics.
 *
 * <p>The JSON form ({@link #toJson()}, {@link #fromJson}) is the {@code normalizer} object of
 * tokenizer.json, so a normalizer read from a Hugging Face file and its canonical JSON are the
 * same value. An empty {@code Sequence} is the identity.</p>
 */
public final class Normalizer {

    private static final String DEFAULT_SOURCE = "tokenizer normalizer";

    /** The normalizer types. */
    public enum Kind {
        NFC("NFC"), NFD("NFD"), NFKC("NFKC"), NFKD("NFKD"), LOWERCASE("Lowercase"), STRIP_ACCENTS("StripAccents"),
        STRIP("Strip"), REPLACE("Replace"), PREPEND("Prepend"), PRECOMPILED("Precompiled"), BERT("BertNormalizer"),
        NMT("Nmt"), BYTE_LEVEL("ByteLevel"), SEQUENCE("Sequence");

        private final String jsonType;

        Kind(String jsonType) {
            this.jsonType = jsonType;
        }

        /** The tokenizer.json {@code type} value. */
        public String jsonType() { return jsonType; }

        static Kind ofJsonType(String type, JsonView v) {
            for (Kind k : values()) {
                if (k.jsonType.equals(type)) return k;
            }
            throw new GgufFormatException(v.where("type") + ": tokenizer normalizer type '" + type
                    + "' is not supported");
        }
    }

    private static final Normalizer IDENTITY = new Normalizer(Kind.SEQUENCE, Params.NONE, Collections.emptyList());

    private final Kind kind;
    private final Params params;
    private final List<Normalizer> children;

    /** The parameters of the kinds that have any; fields a kind does not use hold their defaults. Immutable. */
    private static final class Params {
        static final Params NONE = new Params(false, false, null, false, null, null, null, false, false, null, false);

        final boolean stripLeft;
        final boolean stripRight;
        final String pattern;
        final boolean regex;
        /** Derived from {@link #pattern} and {@link #regex}; not part of equality. */
        final Pattern compiled;
        final String content;
        final PrecompiledCharsmap charsmap;
        final boolean cleanText;
        final boolean handleChineseChars;
        final Boolean stripAccents;
        final boolean lowercase;

        private Params(boolean stripLeft, boolean stripRight, String pattern, boolean regex, Pattern compiled,
                       String content, PrecompiledCharsmap charsmap, boolean cleanText, boolean handleChineseChars,
                       Boolean stripAccents, boolean lowercase) {
            this.stripLeft = stripLeft;
            this.stripRight = stripRight;
            this.pattern = pattern;
            this.regex = regex;
            this.compiled = compiled;
            this.content = content;
            this.charsmap = charsmap;
            this.cleanText = cleanText;
            this.handleChineseChars = handleChineseChars;
            this.stripAccents = stripAccents;
            this.lowercase = lowercase;
        }

        static Params strip(boolean left, boolean right) {
            return new Params(left, right, null, false, null, null, null, false, false, null, false);
        }

        static Params replace(String pattern, boolean regex, Pattern compiled, String content) {
            return new Params(false, false, pattern, regex, compiled, content, null, false, false, null, false);
        }

        static Params content(String content) {
            return new Params(false, false, null, false, null, content, null, false, false, null, false);
        }

        static Params charsmap(PrecompiledCharsmap charsmap) {
            return new Params(false, false, null, false, null, null, charsmap, false, false, null, false);
        }

        static Params bert(boolean cleanText, boolean handleChineseChars, Boolean stripAccents, boolean lowercase) {
            return new Params(false, false, null, false, null, null, null, cleanText, handleChineseChars, stripAccents,
                    lowercase);
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof Params)) return false;
            Params b = (Params) o;
            return stripLeft == b.stripLeft && stripRight == b.stripRight && Objects.equals(pattern, b.pattern)
                    && regex == b.regex && Objects.equals(content, b.content) && Objects.equals(charsmap, b.charsmap)
                    && cleanText == b.cleanText && handleChineseChars == b.handleChineseChars
                    && Objects.equals(stripAccents, b.stripAccents) && lowercase == b.lowercase;
        }

        @Override
        public int hashCode() {
            return Objects.hash(stripLeft, stripRight, pattern, regex, content, charsmap, cleanText, handleChineseChars,
                    stripAccents, lowercase);
        }
    }

    private Normalizer(Kind kind, Params params, List<Normalizer> children) {
        this.kind = kind;
        this.params = params;
        this.children = children;
    }

    private static Normalizer of(Kind kind) {
        return new Normalizer(kind, Params.NONE, Collections.emptyList());
    }

    public static Normalizer identity() { return IDENTITY; }
    public static Normalizer nfc() { return of(Kind.NFC); }
    public static Normalizer nfd() { return of(Kind.NFD); }
    public static Normalizer nfkc() { return of(Kind.NFKC); }
    public static Normalizer nfkd() { return of(Kind.NFKD); }
    /** Full Unicode lowercase per character (no final-sigma context), as Rust {@code char::to_lowercase}. */
    public static Normalizer lowercase() { return of(Kind.LOWERCASE); }
    /** Removes combining marks (categories Mn, Mc, Me); usually after NFD. */
    public static Normalizer stripAccents() { return of(Kind.STRIP_ACCENTS); }
    public static Normalizer nmt() { return of(Kind.NMT); }
    /** GPT-2 byte-to-unicode mapping of the UTF-8 bytes. */
    public static Normalizer byteLevel() { return of(Kind.BYTE_LEVEL); }

    /** Trims Unicode whitespace at the left and/or right end. */
    public static Normalizer strip(boolean left, boolean right) {
        return new Normalizer(Kind.STRIP, Params.strip(left, right), Collections.emptyList());
    }

    /** Replaces every occurrence of the literal {@code pattern} with {@code content}. */
    public static Normalizer replace(String pattern, String content) {
        return replace(pattern, false, content);
    }

    /**
     * Replaces every match of the Oniguruma {@code regex} with {@code content}.
     *
     * @throws IllegalArgumentException when the regex is invalid
     */
    public static Normalizer replaceRegex(String regex, String content) {
        return replace(regex, true, content);
    }

    private static Normalizer replace(String pattern, boolean regex, String content) {
        if (pattern == null) throw new IllegalArgumentException("Replace pattern is null");
        if (content == null) throw new IllegalArgumentException("Replace content is null");
        Pattern compiled = UnicodeRegexSplit.compileOniguruma(regex ? pattern : PreTokenizerSpec.escapeLiteral(pattern));
        return new Normalizer(Kind.REPLACE, Params.replace(pattern, regex, compiled, content), Collections.emptyList());
    }

    /** Prepends {@code prefix} to a non-empty text. */
    public static Normalizer prepend(String prefix) {
        if (prefix == null) throw new IllegalArgumentException("Prepend text is null");
        return new Normalizer(Kind.PREPEND, Params.content(prefix), Collections.emptyList());
    }

    /**
     * The Hugging Face {@code Precompiled} normalizer over a SentencePiece charsmap.
     *
     * @throws IllegalArgumentException when the charsmap's replacement strings are not UTF-8
     */
    public static Normalizer precompiled(PrecompiledCharsmap charsmap) {
        if (charsmap == null) throw new IllegalArgumentException("Precompiled charsmap is null");
        charsmap.requireUtf8Replacements();
        return new Normalizer(Kind.PRECOMPILED, Params.charsmap(charsmap), Collections.emptyList());
    }

    /**
     * {@code BertNormalizer}; {@code stripAccents} null means "as {@code lowercase}".
     */
    public static Normalizer bert(boolean cleanText, boolean handleChineseChars, Boolean stripAccents,
                                  boolean lowercase) {
        return new Normalizer(Kind.BERT, Params.bert(cleanText, handleChineseChars, stripAccents, lowercase),
                Collections.emptyList());
    }

    /** The normalizers applied in order. */
    public static Normalizer sequence(List<Normalizer> normalizers) {
        if (normalizers == null) throw new IllegalArgumentException("Sequence normalizers are null");
        List<Normalizer> copy = new ArrayList<>(normalizers.size());
        for (int i = 0; i < normalizers.size(); i++) {
            Normalizer n = normalizers.get(i);
            if (n == null) throw new IllegalArgumentException("Sequence normalizer " + i + " is null");
            copy.add(n);
        }
        return new Normalizer(Kind.SEQUENCE, Params.NONE, Collections.unmodifiableList(copy));
    }

    public Kind kind() { return kind; }

    /** The members of a {@link Kind#SEQUENCE}; empty otherwise. */
    public List<Normalizer> children() { return children; }

    /** True when the chain changes no text: an empty sequence, or a sequence of such. */
    public boolean isIdentity() {
        if (kind != Kind.SEQUENCE) return false;
        for (Normalizer n : children) {
            if (!n.isIdentity()) return false;
        }
        return true;
    }

    /**
     * Normalizes one text segment.
     *
     * @throws IllegalStateException when a Replace regex exceeds its step budget
     */
    public String normalize(String text) {
        if (text == null) throw new IllegalArgumentException("text is null");
        switch (kind) {
            case NFC: return java.text.Normalizer.normalize(text, java.text.Normalizer.Form.NFC);
            case NFD: return java.text.Normalizer.normalize(text, java.text.Normalizer.Form.NFD);
            case NFKC: return java.text.Normalizer.normalize(text, java.text.Normalizer.Form.NFKC);
            case NFKD: return java.text.Normalizer.normalize(text, java.text.Normalizer.Form.NFKD);
            case LOWERCASE: return lowercase(text);
            case STRIP_ACCENTS: return stripMarks(text, false);
            case STRIP: return strip(text, params.stripLeft, params.stripRight);
            case REPLACE: return replaceAll(params.compiled, params.pattern, params.content, text);
            case PREPEND: return text.isEmpty() ? text : params.content + text;
            case PRECOMPILED: return params.charsmap.normalize(text);
            case BERT: return bertNormalize(text);
            case NMT: return nmt(text);
            case BYTE_LEVEL: return BpeTokenizer.byteEncode(text);
            case SEQUENCE: {
                String s = text;
                for (Normalizer n : children) s = n.normalize(s);
                return s;
            }
            default:
                throw new IllegalStateException("unhandled normalizer kind " + kind);
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Character operations (Hugging Face tokenizers normalizers)
    // ---------------------------------------------------------------------------------------------

    /** Rust {@code char::to_lowercase}: the full unconditional mapping (U+0130 expands). */
    static String lowercase(String text) {
        StringBuilder sb = new StringBuilder(text.length());
        for (int i = 0; i < text.length(); ) {
            int cp = text.codePointAt(i);
            i += Character.charCount(cp);
            if (cp == 0x0130) sb.append("i\u0307");
            else sb.appendCodePoint(Character.toLowerCase(cp));
        }
        return sb.toString();
    }

    /** Removes marks: every M category, or only Mn when {@code nonspacingOnly}. */
    private static String stripMarks(String text, boolean nonspacingOnly) {
        StringBuilder sb = new StringBuilder(text.length());
        for (int i = 0; i < text.length(); ) {
            int cp = text.codePointAt(i);
            i += Character.charCount(cp);
            int type = Character.getType(cp);
            boolean mark = type == Character.NON_SPACING_MARK
                    || (!nonspacingOnly && (type == Character.COMBINING_SPACING_MARK || type == Character.ENCLOSING_MARK));
            if (!mark) sb.appendCodePoint(cp);
        }
        return sb.toString();
    }

    private static String strip(String text, boolean left, boolean right) {
        int start = 0;
        int end = text.length();
        if (left) {
            while (start < end) {
                int cp = text.codePointAt(start);
                if (!UnicodeRegexSplit.isWhiteSpace(cp)) break;
                start += Character.charCount(cp);
            }
        }
        if (right) {
            while (end > start) {
                int cp = text.codePointBefore(end);
                if (!UnicodeRegexSplit.isWhiteSpace(cp)) break;
                end -= Character.charCount(cp);
            }
        }
        return text.substring(start, end);
    }

    /** {@code NormalizedString::replace}: every Hugging Face match becomes {@code content}. */
    static String replaceAll(Pattern compiled, String source, String content, String text) {
        int[] matches = UnicodeRegexSplit.findMatches(compiled, source, text);
        if (matches.length == 0) return text;
        StringBuilder sb = new StringBuilder(text.length() + 16);
        int last = 0;
        for (int i = 0; i < matches.length; i += 2) {
            sb.append(text, last, matches[i]).append(content);
            last = matches[i + 1];
        }
        return sb.append(text, last, text.length()).toString();
    }

    private String bertNormalize(String text) {
        String s = text;
        if (params.cleanText) s = bertClean(s);
        if (params.handleChineseChars) s = spaceChinese(s);
        boolean strip = params.stripAccents != null ? params.stripAccents : params.lowercase;
        if (strip) {
            s = stripMarks(java.text.Normalizer.normalize(s, java.text.Normalizer.Form.NFD), true);
        }
        if (params.lowercase) s = lowercase(s);
        return s;
    }

    /**
     * BertNormalizer {@code clean_text}: NUL, U+FFFD and the other-category characters except tab,
     * LF and CR are dropped first, then each remaining whitespace character becomes a space. The
     * order matters for whitespace controls such as VT, FF and U+0085, which are dropped.
     */
    private static String bertClean(String text) {
        StringBuilder sb = new StringBuilder(text.length());
        for (int i = 0; i < text.length(); ) {
            int cp = text.codePointAt(i);
            i += Character.charCount(cp);
            boolean tabOrLineBreak = cp == '\t' || cp == '\n' || cp == '\r';
            if (cp == 0 || cp == 0xFFFD || (!tabOrLineBreak && isOther(cp))) continue;
            if (tabOrLineBreak || UnicodeRegexSplit.isWhiteSpace(cp)) sb.append(' ');
            else sb.appendCodePoint(cp);
        }
        return sb.toString();
    }

    /** Rust {@code is_other}: general categories Cc, Cf, Cs, Co and Cn. */
    private static boolean isOther(int cp) {
        switch (Character.getType(cp)) {
            case Character.CONTROL:
            case Character.FORMAT:
            case Character.SURROGATE:
            case Character.PRIVATE_USE:
            case Character.UNASSIGNED:
                return true;
            default:
                return false;
        }
    }

    /** BertNormalizer {@code handle_chinese_chars}: spaces around every CJK ideograph. */
    private static String spaceChinese(String text) {
        StringBuilder sb = new StringBuilder(text.length() + 16);
        for (int i = 0; i < text.length(); ) {
            int cp = text.codePointAt(i);
            i += Character.charCount(cp);
            if (isChineseChar(cp)) sb.append(' ').appendCodePoint(cp).append(' ');
            else sb.appendCodePoint(cp);
        }
        return sb.toString();
    }

    /** The Hugging Face BertNormalizer CJK ranges. */
    static boolean isChineseChar(int cp) {
        return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0x20000 && cp <= 0x2A6DF)
                || (cp >= 0x2A700 && cp <= 0x2B73F) || (cp >= 0x2B740 && cp <= 0x2B81F)
                || (cp >= 0x2B920 && cp <= 0x2CEAF) || (cp >= 0xF900 && cp <= 0xFAFF)
                || (cp >= 0x2F800 && cp <= 0x2FA1F);
    }

    /** Nmt: drop ASCII control characters, map the listed separators and U+2581 to a space. */
    private static String nmt(String text) {
        StringBuilder sb = new StringBuilder(text.length());
        for (int i = 0; i < text.length(); ) {
            int cp = text.codePointAt(i);
            i += Character.charCount(cp);
            if ((cp >= 0x01 && cp <= 0x08) || cp == 0x0B || (cp >= 0x0E && cp <= 0x1F) || cp == 0x7F || cp == 0x8F
                    || cp == 0x9F) {
                continue;
            }
            boolean space = cp == 0x09 || cp == 0x0A || cp == 0x0C || cp == 0x0D || cp == 0x1680
                    || (cp >= 0x200B && cp <= 0x200F) || cp == 0x2028 || cp == 0x2029 || cp == 0x2581
                    || cp == 0xFEFF || cp == 0xFFFD;
            if (space) sb.append(' ');
            else sb.appendCodePoint(cp);
        }
        return sb.toString();
    }

    // ---------------------------------------------------------------------------------------------
    // JSON
    // ---------------------------------------------------------------------------------------------

    /** The tokenizer.json {@code normalizer} object of this chain. */
    public Map<String, Object> toJson() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("type", kind.jsonType);
        switch (kind) {
            case STRIP:
                m.put("strip_left", params.stripLeft);
                m.put("strip_right", params.stripRight);
                break;
            case REPLACE:
                m.put("pattern", Collections.singletonMap(params.regex ? "Regex" : "String", params.pattern));
                m.put("content", params.content);
                break;
            case PREPEND:
                m.put("prepend", params.content);
                break;
            case PRECOMPILED:
                m.put("precompiled_charsmap", Base64.getEncoder().encodeToString(params.charsmap.bytes()));
                break;
            case BERT:
                m.put("clean_text", params.cleanText);
                m.put("handle_chinese_chars", params.handleChineseChars);
                m.put("strip_accents", params.stripAccents);
                m.put("lowercase", params.lowercase);
                break;
            case SEQUENCE:
                List<Object> list = new ArrayList<>(children.size());
                for (Normalizer n : children) list.add(n.toJson());
                m.put("normalizers", list);
                break;
            default:
                break;
        }
        return m;
    }

    /** {@link #toJson()} in {@link Json.Style#CANONICAL} form. */
    public String toCanonicalJson() {
        return Json.toString(toJson(), Json.Style.CANONICAL);
    }

    /**
     * Reads {@link #toCanonicalJson()} output, or any tokenizer.json {@code normalizer} object
     * as text.
     *
     * @throws GgufFormatException when the text is not a valid normalizer
     */
    public static Normalizer fromCanonicalJson(String json) {
        if (json == null) throw new GgufFormatException("Invalid " + DEFAULT_SOURCE + ": text is null");
        Object parsed;
        try {
            parsed = Json.parse(json);
        } catch (JsonException e) {
            throw new GgufFormatException("Invalid " + DEFAULT_SOURCE + ": " + e.getMessage(), e);
        }
        return fromJson(parsed, DEFAULT_SOURCE);
    }

    /**
     * Reads a parsed tokenizer.json {@code normalizer} object; {@code source} names it in
     * messages. JSON {@code null} is the identity. Unknown types and members are rejected.
     *
     * @throws GgufFormatException when the value is not a valid normalizer
     */
    public static Normalizer fromJson(Object parsed, String source) {
        String name = source != null ? source : DEFAULT_SOURCE;
        if (parsed == null) return IDENTITY;
        try {
            return read(JsonView.of(parsed, name));
        } catch (JsonException | IllegalArgumentException e) {
            if (e instanceof GgufFormatException) throw (GgufFormatException) e;
            throw new GgufFormatException("Invalid " + name + ": " + e.getMessage(), e);
        }
    }

    private static Normalizer read(JsonView v) {
        Kind kind = Kind.ofJsonType(v.str("type"), v);
        switch (kind) {
            case STRIP:
                allowKeys(v, "strip_left", "strip_right");
                return strip(v.bool("strip_left"), v.bool("strip_right"));
            case REPLACE: {
                allowKeys(v, "pattern", "content");
                JsonView pattern = v.obj("pattern");
                boolean regex = readPatternKind(pattern);
                return replace(pattern.str(regex ? "Regex" : "String"), regex, v.str("content"));
            }
            case PREPEND:
                allowKeys(v, "prepend");
                return prepend(v.str("prepend"));
            case PRECOMPILED: {
                allowKeys(v, "precompiled_charsmap");
                byte[] blob;
                try {
                    blob = Base64.getDecoder().decode(v.str("precompiled_charsmap"));
                } catch (IllegalArgumentException e) {
                    throw new JsonException(v.where("precompiled_charsmap") + ": not valid base64: " + e.getMessage());
                }
                return precompiled(new PrecompiledCharsmap(blob));
            }
            case BERT: {
                allowKeys(v, "clean_text", "handle_chinese_chars", "strip_accents", "lowercase");
                Boolean stripAccents = v.raw("strip_accents") == null ? null : v.bool("strip_accents");
                return bert(v.optBool("clean_text", true), v.optBool("handle_chinese_chars", true), stripAccents,
                        v.optBool("lowercase", true));
            }
            case SEQUENCE: {
                allowKeys(v, "normalizers");
                List<Object> raw = v.list("normalizers");
                List<Normalizer> list = new ArrayList<>(raw.size());
                String base = v.path().isEmpty() ? "normalizers" : v.path() + ".normalizers";
                for (int i = 0; i < raw.size(); i++) {
                    list.add(read(JsonView.of(raw.get(i), v.source(), base + "[" + i + "]")));
                }
                return sequence(list);
            }
            default:
                allowKeys(v);
                return of(kind);
        }
    }

    /** True for a {@code {"Regex": ...}} pattern, false for {@code {"String": ...}}. */
    static boolean readPatternKind(JsonView pattern) {
        if (pattern.keys().size() != 1 || !(pattern.has("Regex") || pattern.has("String"))) {
            throw new JsonException(pattern.source() + ":" + pattern.path()
                    + ": expected exactly one of 'String' or 'Regex', got " + pattern.keys());
        }
        return pattern.has("Regex");
    }

    /** Rejects members other than {@code type} and {@code allowed}. */
    static void allowKeys(JsonView v, String... allowed) {
        Set<String> known = new HashSet<>(Arrays.asList(allowed));
        known.add("type");
        for (String k : v.keys()) {
            if (!known.contains(k)) throw new JsonException(v.where(k) + ": unknown member");
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Normalizer)) return false;
        Normalizer n = (Normalizer) o;
        return kind == n.kind && params.equals(n.params) && children.equals(n.children);
    }

    @Override
    public int hashCode() {
        return Objects.hash(kind, params, children);
    }

    @Override
    public String toString() {
        return kind == Kind.PRECOMPILED ? "Normalizer{Precompiled, " + params.charsmap + "}" : "Normalizer" + toJson();
    }
}

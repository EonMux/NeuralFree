package com.eonmux.neuralfree.gguf.tokenizer;

import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * An immutable description of a BPE pre-tokenizer: an ordered list of {@link Step}s plus the
 * GPT-2 byte-level flags. Registry names are pre-built specs of llama.cpp regex steps
 * ({@link #ofRegexes}); Hugging Face {@code pre_tokenizer} configurations are expressed with the
 * other step kinds and the {@link Dialect#ONIGURUMA} regex dialect.
 *
 * <p>Steps run in order on the current words (initially the whole text), as the Hugging Face
 * {@code Sequence} pre-tokenizer does: each word is processed on its own, empty words are dropped
 * and {@link Behaviour#REMOVED} pieces disappear. {@link Kind#METASPACE} and
 * {@link Kind#BYTE_LEVEL} rewrite the words they receive.</p>
 *
 * <p>{@link #byteEncode()} means the final words are GPT-2 byte-encoded before merging. A
 * {@link Kind#BYTE_LEVEL} step requires it and is configured by {@link #addPrefixSpace()},
 * {@link #useRegex()} and {@link #trimOffsets()}; those flags are false in a spec without one.
 * Unused step fields hold their defaults, and {@link Behaviour#ISOLATED} steps never invert (the
 * flag has no effect there), so equal behaviour gives equal specs and equal canonical JSON.</p>
 */
public final class PreTokenizerSpec {

    /** The most steps a spec may hold. */
    public static final int MAX_STEPS = 1024;

    private static final String DEFAULT_SOURCE = "pre-tokenizer spec";

    /** What a step does. */
    public enum Kind {
        /** Split by {@link Step#pattern()} with a {@link Behaviour}, optionally inverted (HF {@code Split}). */
        REGEX_SPLIT,
        /** HF {@code Whitespace}: keep {@code \w+|[^\w\s]+} runs, drop the rest. */
        WHITESPACE,
        /** HF {@code WhitespaceSplit}: drop every Unicode whitespace character, keep the runs between. */
        WHITESPACE_SPLIT,
        /** HF {@code Punctuation}: split at each ASCII or Unicode punctuation character with a behaviour. */
        PUNCTUATION,
        /** HF {@code CharDelimiterSplit}: drop each {@link Step#delimiter()} character. */
        CHAR_DELIMITER,
        /** HF {@code Digits}: isolate numeric characters, each one or in runs. */
        DIGITS,
        /** HF {@code ByteLevel}: prefix space, GPT-2 split and byte-level mapping, per the spec flags. */
        BYTE_LEVEL,
        /** HF {@code Metaspace}: spaces become the replacement, which is prepended and split before. */
        METASPACE,
        /** HF {@code UnicodeScripts}: split where the Unicode script changes. */
        UNICODE_SCRIPTS,
        /** HF {@code BertPreTokenizer}: drop whitespace, then isolate punctuation. */
        BERT
    }

    /** How a {@link Kind#REGEX_SPLIT} pattern is read. */
    public enum Dialect {
        /**
         * llama.cpp {@code unicode_regex_split}: hand-written splitters by exact text, Unicode
         * categories collapsed, ASCII-only {@code \s\S\d}.
         */
        LLAMA_CPP,
        /**
         * Oniguruma as used by Hugging Face tokenizers: Unicode {@code \s\w\d}, real
         * sub-categories, script properties such as {@code \p{Han}} and line anchors.
         */
        ONIGURUMA
    }

    /** HF {@code SplitDelimiterBehavior}: what happens to the matched pieces. */
    public enum Behaviour {
        /** Each match and each gap is a word. */
        ISOLATED,
        /** Matches are dropped. */
        REMOVED,
        /** A match joins the gap before it. */
        MERGED_WITH_PREVIOUS,
        /** A match joins the gap after it. */
        MERGED_WITH_NEXT,
        /** Consecutive matches form one word. */
        CONTIGUOUS
    }

    /** HF Metaspace {@code prepend_scheme}: when the replacement is prepended to a word. */
    public enum PrependScheme {
        /** To every word that does not start with it. */
        ALWAYS,
        /** Only to a word at the very start of the input. */
        FIRST,
        /** Never. */
        NEVER
    }

    /** One immutable pre-tokenizer step; build one with the static factories. */
    public static final class Step {
        private final Kind kind;
        private final String pattern;
        private final Dialect dialect;
        private final Behaviour behaviour;
        private final boolean invert;
        private final boolean individualDigits;
        private final int delimiter;
        private final int replacement;
        private final PrependScheme prependScheme;
        private final boolean split;

        private Step(Kind kind, String pattern, Dialect dialect, Behaviour behaviour, boolean invert,
                     boolean individualDigits, int delimiter, int replacement, PrependScheme prependScheme,
                     boolean split) {
            this.kind = kind;
            this.pattern = pattern;
            this.dialect = dialect;
            this.behaviour = behaviour;
            this.invert = invert && behaviour != Behaviour.ISOLATED;
            this.individualDigits = individualDigits;
            this.delimiter = delimiter;
            this.replacement = replacement;
            this.prependScheme = prependScheme;
            this.split = split;
        }

        private static Step plain(Kind kind) {
            return new Step(kind, null, null, null, false, false, -1, -1, null, false);
        }

        /** A regex split; {@code invert} flips which pieces count as matches. */
        public static Step regex(String pattern, Dialect dialect, Behaviour behaviour, boolean invert) {
            if (pattern == null) throw new IllegalArgumentException("regex split pattern is null");
            if (dialect == null) throw new IllegalArgumentException("regex split dialect is null");
            if (behaviour == null) throw new IllegalArgumentException("regex split behaviour is null");
            return new Step(Kind.REGEX_SPLIT, pattern, dialect, behaviour, invert, false, -1, -1, null, false);
        }

        /** A split on a literal string (HF {@code Split} with a {@code String} pattern). */
        public static Step literal(String text, Behaviour behaviour, boolean invert) {
            if (text == null) throw new IllegalArgumentException("literal split text is null");
            return regex(escapeLiteral(text), Dialect.ONIGURUMA, behaviour, invert);
        }

        public static Step whitespace() { return plain(Kind.WHITESPACE); }

        public static Step whitespaceSplit() { return plain(Kind.WHITESPACE_SPLIT); }

        public static Step punctuation(Behaviour behaviour) {
            if (behaviour == null) throw new IllegalArgumentException("punctuation behaviour is null");
            return new Step(Kind.PUNCTUATION, null, null, behaviour, false, false, -1, -1, null, false);
        }

        /** Splits at, and drops, every occurrence of the code point {@code delimiter}. */
        public static Step charDelimiter(int delimiter) {
            requireCodePoint(delimiter, "delimiter");
            return new Step(Kind.CHAR_DELIMITER, null, null, null, false, false, delimiter, -1, null, false);
        }

        public static Step digits(boolean individualDigits) {
            return new Step(Kind.DIGITS, null, null, null, false, individualDigits, -1, -1, null, false);
        }

        public static Step byteLevel() { return plain(Kind.BYTE_LEVEL); }

        /**
         * HF Metaspace with the code point {@code replacement}; with {@code split} each word is
         * split before every replacement character.
         */
        public static Step metaspace(int replacement, PrependScheme prependScheme, boolean split) {
            requireCodePoint(replacement, "replacement");
            if (prependScheme == null) throw new IllegalArgumentException("metaspace prepend scheme is null");
            return new Step(Kind.METASPACE, null, null, null, false, false, -1, replacement, prependScheme, split);
        }

        public static Step unicodeScripts() { return plain(Kind.UNICODE_SCRIPTS); }

        public static Step bert() { return plain(Kind.BERT); }

        public Kind kind() { return kind; }
        /** The regex text of a {@link Kind#REGEX_SPLIT} step, else null. */
        public String pattern() { return pattern; }
        /** The regex dialect of a {@link Kind#REGEX_SPLIT} step, else null. */
        public Dialect dialect() { return dialect; }
        /** The behaviour of a {@link Kind#REGEX_SPLIT} or {@link Kind#PUNCTUATION} step, else null. */
        public Behaviour behaviour() { return behaviour; }
        /** Matches and gaps swap roles (never true for {@link Behaviour#ISOLATED}). */
        public boolean invert() { return invert; }
        /** {@link Kind#DIGITS}: each digit is its own word instead of runs. */
        public boolean individualDigits() { return individualDigits; }
        /** The {@link Kind#CHAR_DELIMITER} code point, else -1. */
        public int delimiter() { return delimiter; }
        /** The {@link Kind#METASPACE} replacement code point, else -1. */
        public int replacement() { return replacement; }
        /** The {@link Kind#METASPACE} prepend scheme, else null. */
        public PrependScheme prependScheme() { return prependScheme; }
        /** {@link Kind#METASPACE}: words are split before each replacement character. */
        public boolean split() { return split; }

        Map<String, Object> toJson() {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("kind", kind.name());
            switch (kind) {
                case REGEX_SPLIT:
                    m.put("pattern", pattern);
                    m.put("dialect", dialect.name());
                    m.put("behaviour", behaviour.name());
                    m.put("invert", invert);
                    break;
                case PUNCTUATION:
                    m.put("behaviour", behaviour.name());
                    break;
                case CHAR_DELIMITER:
                    m.put("delimiter", new String(Character.toChars(delimiter)));
                    break;
                case DIGITS:
                    m.put("individual_digits", individualDigits);
                    break;
                case METASPACE:
                    m.put("replacement", new String(Character.toChars(replacement)));
                    m.put("prepend_scheme", prependScheme.name());
                    m.put("split", split);
                    break;
                default:
                    break;
            }
            return m;
        }

        static Step fromJson(JsonView v) {
            Kind kind = enumValue(Kind.class, v, "kind");
            switch (kind) {
                case REGEX_SPLIT:
                    requireKeys(v, "kind", "pattern", "dialect", "behaviour", "invert");
                    return regex(v.str("pattern"), enumValue(Dialect.class, v, "dialect"),
                            enumValue(Behaviour.class, v, "behaviour"), v.bool("invert"));
                case PUNCTUATION:
                    requireKeys(v, "kind", "behaviour");
                    return punctuation(enumValue(Behaviour.class, v, "behaviour"));
                case CHAR_DELIMITER:
                    requireKeys(v, "kind", "delimiter");
                    return charDelimiter(singleCodePoint(v, "delimiter"));
                case DIGITS:
                    requireKeys(v, "kind", "individual_digits");
                    return digits(v.bool("individual_digits"));
                case METASPACE:
                    requireKeys(v, "kind", "replacement", "prepend_scheme", "split");
                    return metaspace(singleCodePoint(v, "replacement"),
                            enumValue(PrependScheme.class, v, "prepend_scheme"), v.bool("split"));
                default:
                    requireKeys(v, "kind");
                    return plain(kind);
            }
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof Step)) return false;
            Step s = (Step) o;
            return kind == s.kind && Objects.equals(pattern, s.pattern) && dialect == s.dialect
                    && behaviour == s.behaviour && invert == s.invert && individualDigits == s.individualDigits
                    && delimiter == s.delimiter && replacement == s.replacement
                    && prependScheme == s.prependScheme && split == s.split;
        }

        @Override
        public int hashCode() {
            return Objects.hash(kind, pattern, dialect, behaviour, invert, individualDigits, delimiter, replacement,
                    prependScheme, split);
        }

        @Override
        public String toString() {
            return toJson().toString();
        }
    }

    private final List<Step> steps;
    private final boolean byteEncode;
    private final boolean addPrefixSpace;
    private final boolean trimOffsets;
    private final boolean useRegex;

    /**
     * @param steps          the steps in order (copied)
     * @param byteEncode     the final words are GPT-2 byte-encoded before merging
     * @param addPrefixSpace {@link Kind#BYTE_LEVEL}: a space is prepended to each word lacking one
     * @param trimOffsets    {@link Kind#BYTE_LEVEL}: offsets exclude surrounding whitespace (no effect on ids)
     * @param useRegex       {@link Kind#BYTE_LEVEL}: words are split with the GPT-2 regex
     * @throws IllegalArgumentException when the steps or flags are inconsistent
     */
    public PreTokenizerSpec(List<Step> steps, boolean byteEncode, boolean addPrefixSpace, boolean trimOffsets,
                            boolean useRegex) {
        if (steps == null) throw new IllegalArgumentException("pre-tokenizer steps are null");
        if (steps.size() > MAX_STEPS) {
            throw new IllegalArgumentException("pre-tokenizer has " + steps.size() + " steps, more than " + MAX_STEPS);
        }
        List<Step> copy = new ArrayList<>(steps.size());
        int byteLevel = 0;
        for (int i = 0; i < steps.size(); i++) {
            Step s = steps.get(i);
            if (s == null) throw new IllegalArgumentException("pre-tokenizer step " + i + " is null");
            if (s.kind == Kind.BYTE_LEVEL) byteLevel++;
            copy.add(s);
        }
        if (byteLevel > 1) {
            throw new IllegalArgumentException("pre-tokenizer has " + byteLevel + " BYTE_LEVEL steps; at most one is"
                    + " allowed, since a second one would byte-encode twice");
        }
        if (byteLevel == 1 && !byteEncode) {
            throw new IllegalArgumentException("pre-tokenizer with a BYTE_LEVEL step must set byteEncode");
        }
        if (byteLevel == 0 && (addPrefixSpace || trimOffsets || useRegex)) {
            throw new IllegalArgumentException("pre-tokenizer flags addPrefixSpace, trimOffsets and useRegex"
                    + " configure a BYTE_LEVEL step, and this spec has none");
        }
        this.steps = Collections.unmodifiableList(copy);
        this.byteEncode = byteEncode;
        this.addPrefixSpace = addPrefixSpace;
        this.trimOffsets = trimOffsets;
        this.useRegex = useRegex;
    }

    /**
     * A spec of llama.cpp regex steps, each {@link Behaviour#ISOLATED} and not inverted: the form
     * of every registry name.
     */
    public static PreTokenizerSpec ofRegexes(List<String> regexes, boolean byteEncode) {
        if (regexes == null) throw new IllegalArgumentException("regexes are null");
        List<Step> steps = new ArrayList<>(regexes.size());
        for (String r : regexes) steps.add(Step.regex(r, Dialect.LLAMA_CPP, Behaviour.ISOLATED, false));
        return new PreTokenizerSpec(steps, byteEncode, false, false, false);
    }

    public List<Step> steps() { return steps; }
    /** The final words are GPT-2 byte-encoded before merging. */
    public boolean byteEncode() { return byteEncode; }
    public boolean addPrefixSpace() { return addPrefixSpace; }
    public boolean trimOffsets() { return trimOffsets; }
    public boolean useRegex() { return useRegex; }

    /** The index of the {@link Kind#BYTE_LEVEL} step, or -1. */
    public int byteLevelIndex() {
        for (int i = 0; i < steps.size(); i++) {
            if (steps.get(i).kind == Kind.BYTE_LEVEL) return i;
        }
        return -1;
    }

    /**
     * True when steps follow the {@link Kind#BYTE_LEVEL} step: they then split byte-level text, so
     * the byte encoding happens inside the split instead of after it.
     */
    public boolean encodesBeforeLastStep() {
        int i = byteLevelIndex();
        return i >= 0 && i < steps.size() - 1;
    }

    /** The spec as a JSON value: {@code steps} plus the four flags, unused step fields omitted. */
    public Map<String, Object> toJson() {
        List<Object> list = new ArrayList<>(steps.size());
        for (Step s : steps) list.add(s.toJson());
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("steps", list);
        m.put("byte_encode", byteEncode);
        m.put("add_prefix_space", addPrefixSpace);
        m.put("trim_offsets", trimOffsets);
        m.put("use_regex", useRegex);
        return m;
    }

    /** {@link #toJson()} in {@link Json.Style#CANONICAL} form: equal specs give equal strings. */
    public String toCanonicalJson() {
        return Json.toString(toJson(), Json.Style.CANONICAL);
    }

    /**
     * Reads {@link #toCanonicalJson()} output.
     *
     * @throws GgufFormatException when the text is not a valid spec
     */
    public static PreTokenizerSpec fromCanonicalJson(String json) {
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
     * Reads a parsed {@link #toJson()} value; {@code source} names it in messages. Every member is
     * required and unknown members are rejected.
     *
     * @throws GgufFormatException when the value is not a valid spec
     */
    public static PreTokenizerSpec fromJson(Object parsed, String source) {
        String name = source != null ? source : DEFAULT_SOURCE;
        try {
            JsonView v = JsonView.of(parsed, name);
            requireKeys(v, "steps", "byte_encode", "add_prefix_space", "trim_offsets", "use_regex");
            List<Object> raw = v.list("steps");
            if (raw.size() > MAX_STEPS) {
                throw new GgufFormatException("Invalid " + name + ": " + raw.size() + " steps, more than " + MAX_STEPS);
            }
            List<Step> steps = new ArrayList<>(raw.size());
            for (int i = 0; i < raw.size(); i++) steps.add(Step.fromJson(JsonView.of(raw.get(i), name, "steps[" + i + "]")));
            return new PreTokenizerSpec(steps, v.bool("byte_encode"), v.bool("add_prefix_space"),
                    v.bool("trim_offsets"), v.bool("use_regex"));
        } catch (JsonException | IllegalArgumentException e) {
            if (e instanceof GgufFormatException) throw (GgufFormatException) e;
            throw new GgufFormatException("Invalid " + name + ": " + e.getMessage(), e);
        }
    }

    /**
     * The Oniguruma (and Java) regex matching {@code text} literally: every regex metacharacter
     * is backslash-escaped, as Hugging Face escapes {@code Split} string patterns.
     */
    static String escapeLiteral(String text) {
        StringBuilder sb = new StringBuilder(text.length() + 8);
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if ("\\^$.|?*+()[]{}-#&~".indexOf(c) >= 0) sb.append('\\');
            sb.append(c);
        }
        return sb.toString();
    }

    private static void requireCodePoint(int cp, String what) {
        if (cp < 0 || cp > Character.MAX_CODE_POINT || (cp >= Character.MIN_SURROGATE && cp <= Character.MAX_SURROGATE)) {
            throw new IllegalArgumentException("pre-tokenizer " + what + " " + cp + " is not a Unicode scalar value");
        }
    }

    private static int singleCodePoint(JsonView v, String key) {
        String s = v.str(key);
        if (s.isEmpty() || s.codePointCount(0, s.length()) != 1) {
            throw new JsonException(v.where(key) + ": expected exactly one character, got '" + s + "'");
        }
        return s.codePointAt(0);
    }

    private static <E extends Enum<E>> E enumValue(Class<E> type, JsonView v, String key) {
        String s = v.str(key);
        try {
            return Enum.valueOf(type, s);
        } catch (IllegalArgumentException e) {
            throw new JsonException(v.where(key) + ": unknown value '" + s + "', expected one of "
                    + Arrays.toString(type.getEnumConstants()));
        }
    }

    private static void requireKeys(JsonView v, String... allowed) {
        Set<String> known = new HashSet<>(Arrays.asList(allowed));
        for (String k : v.keys()) {
            if (!known.contains(k)) throw new JsonException(v.where(k) + ": unknown member");
        }
        for (String k : allowed) {
            if (!v.has(k)) throw new JsonException(v.where(k) + ": required member is missing");
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof PreTokenizerSpec)) return false;
        PreTokenizerSpec s = (PreTokenizerSpec) o;
        return byteEncode == s.byteEncode && addPrefixSpace == s.addPrefixSpace && trimOffsets == s.trimOffsets
                && useRegex == s.useRegex && steps.equals(s.steps);
    }

    @Override
    public int hashCode() {
        return Objects.hash(steps, byteEncode, addPrefixSpace, trimOffsets, useRegex);
    }

    @Override
    public String toString() {
        return "PreTokenizerSpec" + toJson();
    }
}

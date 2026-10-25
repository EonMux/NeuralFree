package com.eonmux.neuralfree.gguf.tokenizer;

import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Pattern;

/**
 * An immutable Hugging Face decoder chain (tokenizers {@code decoders}) turning token pieces
 * back into text: ByteLevel, Metaspace, WordPiece, BPEDecoder, Replace, Strip, ByteFallback, Fuse
 * and Sequence, with Hugging Face's per-token semantics ({@code decode_chain}, then the pieces
 * joined).
 *
 * <p>The JSON form ({@link #toJson()}, {@link #fromJson}) is the {@code decoder} object of
 * tokenizer.json. {@code CTC} is rejected. An empty {@code Sequence} joins the pieces
 * unchanged.</p>
 */
public final class TokenDecoder {

    private static final String DEFAULT_SOURCE = "tokenizer decoder";
    private static final char REPLACEMENT_CHAR = '\uFFFD';

    /** The decoder types. */
    public enum Kind {
        BYTE_LEVEL("ByteLevel"), METASPACE("Metaspace"), WORD_PIECE("WordPiece"), BPE("BPEDecoder"),
        REPLACE("Replace"), STRIP("Strip"), BYTE_FALLBACK("ByteFallback"), FUSE("Fuse"), SEQUENCE("Sequence");

        private final String jsonType;

        Kind(String jsonType) {
            this.jsonType = jsonType;
        }

        /** The tokenizer.json {@code type} value. */
        public String jsonType() { return jsonType; }

        static Kind ofJsonType(String type, JsonView v) {
            if ("CTC".equals(type)) throw new GgufFormatException("tokenizer decoder CTC is not supported");
            for (Kind k : values()) {
                if (k.jsonType.equals(type)) return k;
            }
            throw new GgufFormatException(v.where("type") + ": tokenizer decoder type '" + type + "' is not supported");
        }
    }

    private static final TokenDecoder IDENTITY = new TokenDecoder(Kind.SEQUENCE, Params.NONE, Collections.emptyList());

    private final Kind kind;
    private final Params params;
    private final List<TokenDecoder> children;

    /** The parameters of the kinds that have any; fields a kind does not use hold their defaults. Immutable. */
    private static final class Params {
        static final Params NONE = new Params();

        final boolean addPrefixSpace;
        final boolean trimOffsets;
        final boolean useRegex;
        final int replacement;
        final PreTokenizerSpec.PrependScheme prependScheme;
        final boolean split;
        final String prefix;
        final boolean cleanup;
        final String pattern;
        final boolean regex;
        /** Derived from {@link #pattern} and {@link #regex}; not part of equality. */
        final Pattern compiled;
        final String content;
        final int stripChar;
        final int start;
        final int stop;

        private Params() {
            this(false, false, false, 0, null, false, null, false, null, false, null, null, 0, 0, 0);
        }

        private Params(boolean addPrefixSpace, boolean trimOffsets, boolean useRegex, int replacement,
                       PreTokenizerSpec.PrependScheme prependScheme, boolean split, String prefix, boolean cleanup,
                       String pattern, boolean regex, Pattern compiled, String content, int stripChar, int start,
                       int stop) {
            this.addPrefixSpace = addPrefixSpace;
            this.trimOffsets = trimOffsets;
            this.useRegex = useRegex;
            this.replacement = replacement;
            this.prependScheme = prependScheme;
            this.split = split;
            this.prefix = prefix;
            this.cleanup = cleanup;
            this.pattern = pattern;
            this.regex = regex;
            this.compiled = compiled;
            this.content = content;
            this.stripChar = stripChar;
            this.start = start;
            this.stop = stop;
        }

        static Params byteLevel(boolean addPrefixSpace, boolean trimOffsets, boolean useRegex) {
            return new Params(addPrefixSpace, trimOffsets, useRegex, 0, null, false, null, false, null, false, null,
                    null, 0, 0, 0);
        }

        static Params metaspace(int replacement, PreTokenizerSpec.PrependScheme prependScheme, boolean split) {
            return new Params(false, false, false, replacement, prependScheme, split, null, false, null, false, null,
                    null, 0, 0, 0);
        }

        static Params wordPiece(String prefix, boolean cleanup) {
            return new Params(false, false, false, 0, null, false, prefix, cleanup, null, false, null, null, 0, 0, 0);
        }

        static Params content(String content) {
            return new Params(false, false, false, 0, null, false, null, false, null, false, null, content, 0, 0, 0);
        }

        static Params replace(String pattern, boolean regex, Pattern compiled, String content) {
            return new Params(false, false, false, 0, null, false, null, false, pattern, regex, compiled, content, 0,
                    0, 0);
        }

        static Params strip(int stripChar, int start, int stop) {
            return new Params(false, false, false, 0, null, false, null, false, null, false, null, null, stripChar,
                    start, stop);
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof Params)) return false;
            Params b = (Params) o;
            return addPrefixSpace == b.addPrefixSpace && trimOffsets == b.trimOffsets && useRegex == b.useRegex
                    && replacement == b.replacement && prependScheme == b.prependScheme && split == b.split
                    && Objects.equals(prefix, b.prefix) && cleanup == b.cleanup && Objects.equals(pattern, b.pattern)
                    && regex == b.regex && Objects.equals(content, b.content) && stripChar == b.stripChar
                    && start == b.start && stop == b.stop;
        }

        @Override
        public int hashCode() {
            return Objects.hash(addPrefixSpace, trimOffsets, useRegex, replacement, prependScheme, split, prefix,
                    cleanup, pattern, regex, content, stripChar, start, stop);
        }
    }

    private TokenDecoder(Kind kind, Params params, List<TokenDecoder> children) {
        this.kind = kind;
        this.params = params;
        this.children = children;
    }

    /** An empty Sequence: the pieces joined unchanged. */
    public static TokenDecoder identity() { return IDENTITY; }

    /** GPT-2 byte-level decoding; the flags only round-trip the tokenizer.json object. */
    public static TokenDecoder byteLevel(boolean addPrefixSpace, boolean trimOffsets, boolean useRegex) {
        return new TokenDecoder(Kind.BYTE_LEVEL, Params.byteLevel(addPrefixSpace, trimOffsets, useRegex),
                Collections.emptyList());
    }

    /**
     * Metaspace: the replacement becomes a space, except in the first piece where it is removed
     * unless the scheme is {@link PreTokenizerSpec.PrependScheme#NEVER}; {@code split} only
     * round-trips the tokenizer.json object.
     */
    public static TokenDecoder metaspace(int replacement, PreTokenizerSpec.PrependScheme prependScheme, boolean split) {
        requireScalar(replacement, "Metaspace replacement");
        if (prependScheme == null) throw new IllegalArgumentException("Metaspace prepend scheme is null");
        return new TokenDecoder(Kind.METASPACE, Params.metaspace(replacement, prependScheme, split),
                Collections.emptyList());
    }

    /** WordPiece: continuation pieces lose {@code prefix}, others gain a leading space. */
    public static TokenDecoder wordPiece(String prefix, boolean cleanup) {
        if (prefix == null) throw new IllegalArgumentException("WordPiece prefix is null");
        return new TokenDecoder(Kind.WORD_PIECE, Params.wordPiece(prefix, cleanup), Collections.emptyList());
    }

    /** BPEDecoder: {@code suffix} becomes a space, or nothing in the last piece. */
    public static TokenDecoder bpe(String suffix) {
        if (suffix == null) throw new IllegalArgumentException("BPEDecoder suffix is null");
        return new TokenDecoder(Kind.BPE, Params.content(suffix), Collections.emptyList());
    }

    /** Replaces every occurrence of the literal {@code pattern} in each piece. */
    public static TokenDecoder replace(String pattern, String content) {
        return replace(pattern, false, content);
    }

    /**
     * Replaces every match of the Oniguruma {@code regex} in each piece.
     *
     * @throws IllegalArgumentException when the regex is invalid
     */
    public static TokenDecoder replaceRegex(String regex, String content) {
        return replace(regex, true, content);
    }

    private static TokenDecoder replace(String pattern, boolean regex, String content) {
        if (pattern == null) throw new IllegalArgumentException("Replace pattern is null");
        if (content == null) throw new IllegalArgumentException("Replace content is null");
        Pattern compiled = UnicodeRegexSplit.compileOniguruma(regex ? pattern : PreTokenizerSpec.escapeLiteral(pattern));
        return new TokenDecoder(Kind.REPLACE, Params.replace(pattern, regex, compiled, content), Collections.emptyList());
    }

    /**
     * Strip: removes up to {@code start} leading and {@code stop} trailing {@code content}
     * characters from each piece; a piece whose cuts meet becomes empty.
     */
    public static TokenDecoder strip(int content, int start, int stop) {
        requireScalar(content, "Strip content");
        if (start < 0 || stop < 0) {
            throw new IllegalArgumentException("Strip start " + start + " and stop " + stop + " must not be negative");
        }
        return new TokenDecoder(Kind.STRIP, Params.strip(content, start, stop), Collections.emptyList());
    }

    /** ByteFallback: runs of {@code <0xHH>} pieces become their UTF-8 text, or U+FFFD per byte. */
    public static TokenDecoder byteFallback() {
        return new TokenDecoder(Kind.BYTE_FALLBACK, Params.NONE, Collections.emptyList());
    }

    /** Fuse: all pieces joined into one. */
    public static TokenDecoder fuse() {
        return new TokenDecoder(Kind.FUSE, Params.NONE, Collections.emptyList());
    }

    /** The decoders applied in order. */
    public static TokenDecoder sequence(List<TokenDecoder> decoders) {
        if (decoders == null) throw new IllegalArgumentException("Sequence decoders are null");
        List<TokenDecoder> copy = new ArrayList<>(decoders.size());
        for (int i = 0; i < decoders.size(); i++) {
            TokenDecoder d = decoders.get(i);
            if (d == null) throw new IllegalArgumentException("Sequence decoder " + i + " is null");
            copy.add(d);
        }
        return new TokenDecoder(Kind.SEQUENCE, Params.NONE, Collections.unmodifiableList(copy));
    }

    private static void requireScalar(int cp, String what) {
        if (cp < 0 || cp > Character.MAX_CODE_POINT || (cp >= Character.MIN_SURROGATE && cp <= Character.MAX_SURROGATE)) {
            throw new IllegalArgumentException(what + " " + cp + " is not a Unicode scalar value");
        }
    }

    public Kind kind() { return kind; }

    /** The members of a {@link Kind#SEQUENCE}; empty otherwise. */
    public List<TokenDecoder> children() { return children; }

    /**
     * The text of {@code pieces}: the chain applied, then the resulting pieces joined.
     *
     * @throws IllegalStateException when a Replace regex exceeds its step budget
     */
    public String decode(List<String> pieces) {
        if (pieces == null) throw new IllegalArgumentException("pieces are null");
        for (int i = 0; i < pieces.size(); i++) {
            if (pieces.get(i) == null) throw new IllegalArgumentException("piece " + i + " is null");
        }
        return String.join("", decodeChain(pieces));
    }

    /** Hugging Face {@code decode_chain}: the pieces after this decoder, before joining. */
    public List<String> decodeChain(List<String> pieces) {
        switch (kind) {
            case BYTE_LEVEL: return Collections.singletonList(byteLevelDecode(pieces));
            case METASPACE: return metaspaceDecode(pieces);
            case WORD_PIECE: return wordPieceDecode(pieces);
            case BPE: return bpeDecode(pieces);
            case REPLACE: {
                List<String> out = new ArrayList<>(pieces.size());
                for (String t : pieces) out.add(Normalizer.replaceAll(params.compiled, params.pattern, params.content, t));
                return out;
            }
            case STRIP: return stripDecode(pieces);
            case BYTE_FALLBACK: return byteFallbackDecode(pieces);
            case FUSE: return Collections.singletonList(String.join("", pieces));
            case SEQUENCE: {
                List<String> current = pieces;
                for (TokenDecoder d : children) current = d.decodeChain(current);
                return current;
            }
            default:
                throw new IllegalStateException("unhandled decoder kind " + kind);
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Decoders (Hugging Face tokenizers decoders)
    // ---------------------------------------------------------------------------------------------

    /** Each piece's GPT-2 bytes (its raw UTF-8 when a character is outside the table), decoded lossily. */
    private static String byteLevelDecode(List<String> pieces) {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        for (String t : pieces) {
            byte[] mapped = gpt2Bytes(t);
            byte[] b = mapped != null ? mapped : t.getBytes(StandardCharsets.UTF_8);
            bytes.write(b, 0, b.length);
        }
        return lossyUtf8(bytes.toByteArray());
    }

    /** The GPT-2 bytes of a piece ({@link BpeTokenizer#gpt2Byte}), or null when a character is outside the table. */
    private static byte[] gpt2Bytes(String piece) {
        byte[] out = new byte[piece.length()];
        int n = 0;
        for (int i = 0; i < piece.length(); ) {
            int cp = piece.codePointAt(i);
            i += Character.charCount(cp);
            int b = BpeTokenizer.gpt2Byte(cp);
            if (b < 0) return null;
            out[n++] = (byte) b;
        }
        return Arrays.copyOf(out, n);
    }

    /**
     * Rust {@code String::from_utf8_lossy}: one U+FFFD per maximal subpart of an ill-formed
     * sequence (Unicode 3.9 best practice), so an encoded surrogate yields three.
     */
    static String lossyUtf8(byte[] bytes) {
        StringBuilder sb = new StringBuilder(bytes.length);
        int i = 0;
        while (i < bytes.length) {
            int b0 = bytes[i] & 0xFF;
            if (b0 < 0x80) {
                sb.append((char) b0);
                i++;
                continue;
            }
            int width = b0 >= 0xC2 && b0 <= 0xDF ? 2 : b0 >= 0xE0 && b0 <= 0xEF ? 3 : b0 >= 0xF0 && b0 <= 0xF4 ? 4 : 0;
            int valid = 1;
            while (valid < width && i + valid < bytes.length
                    && inRange(bytes[i + valid] & 0xFF, valid == 1 ? secondLow(b0) : 0x80,
                               valid == 1 ? secondHigh(b0) : 0xBF)) {
                valid++;
            }
            if (width == 0 || valid < width) {
                sb.append(REPLACEMENT_CHAR);
                i += valid;
                continue;
            }
            sb.append(new String(bytes, i, width, StandardCharsets.UTF_8));
            i += width;
        }
        return sb.toString();
    }

    private static boolean inRange(int b, int lo, int hi) {
        return b >= lo && b <= hi;
    }

    /** The lowest valid second byte after lead {@code b0} (excludes overlongs). */
    private static int secondLow(int b0) {
        return b0 == 0xE0 ? 0xA0 : b0 == 0xF0 ? 0x90 : 0x80;
    }

    /** The highest valid second byte after lead {@code b0} (excludes surrogates and values past U+10FFFF). */
    private static int secondHigh(int b0) {
        return b0 == 0xED ? 0x9F : b0 == 0xF4 ? 0x8F : 0xBF;
    }

    private List<String> metaspaceDecode(List<String> pieces) {
        List<String> out = new ArrayList<>(pieces.size());
        boolean dropFirst = params.prependScheme != PreTokenizerSpec.PrependScheme.NEVER;
        for (int i = 0; i < pieces.size(); i++) {
            String t = pieces.get(i);
            StringBuilder sb = new StringBuilder(t.length());
            for (int j = 0; j < t.length(); ) {
                int cp = t.codePointAt(j);
                j += Character.charCount(cp);
                if (cp != params.replacement) sb.appendCodePoint(cp);
                else if (i != 0 || !dropFirst) sb.append(' ');
            }
            out.add(sb.toString());
        }
        return out;
    }

    private List<String> wordPieceDecode(List<String> pieces) {
        List<String> out = new ArrayList<>(pieces.size());
        for (int i = 0; i < pieces.size(); i++) {
            String t = pieces.get(i);
            if (i != 0) {
                if (t.startsWith(params.prefix)) t = replaceFirst(t, params.prefix);
                else t = " " + t;
            }
            out.add(params.cleanup ? cleanup(t) : t);
        }
        return out;
    }

    /** Rust {@code str::replacen(prefix, "", 1)}: an empty prefix leaves the piece unchanged. */
    private static String replaceFirst(String t, String prefix) {
        return prefix.isEmpty() ? t : t.substring(prefix.length());
    }

    /** The WordPiece {@code cleanup} replacements, in Hugging Face's order. */
    static String cleanup(String s) {
        return s.replace(" .", ".").replace(" ?", "?").replace(" !", "!").replace(" ,", ",").replace(" ' ", "'")
                .replace(" n't", "n't").replace(" 'm", "'m").replace(" do not", " don't").replace(" 's", "'s")
                .replace(" 've", "'ve").replace(" 're", "'re");
    }

    private List<String> bpeDecode(List<String> pieces) {
        List<String> out = new ArrayList<>(pieces.size());
        for (int i = 0; i < pieces.size(); i++) {
            String t = pieces.get(i);
            out.add(params.content.isEmpty() ? t : t.replace(params.content, i == pieces.size() - 1 ? "" : " "));
        }
        return out;
    }

    private List<String> stripDecode(List<String> pieces) {
        List<String> out = new ArrayList<>(pieces.size());
        for (String t : pieces) {
            int[] cps = t.codePoints().toArray();
            int startCut = 0;
            while (startCut < Math.min(params.start, cps.length) && cps[startCut] == params.stripChar) startCut++;
            int stopCut = cps.length;
            int limit = Math.max(cps.length - params.stop, 0);
            while (stopCut > limit && cps[stopCut - 1] == params.stripChar) stopCut--;
            out.add(startCut >= stopCut ? "" : new String(cps, startCut, stopCut - startCut));
        }
        return out;
    }

    private static List<String> byteFallbackDecode(List<String> pieces) {
        List<String> out = new ArrayList<>(pieces.size());
        ByteArrayOutputStream run = new ByteArrayOutputStream();
        for (String t : pieces) {
            int b = byteToken(t);
            if (b >= 0) {
                run.write(b);
                continue;
            }
            flushBytes(run, out);
            out.add(t);
        }
        flushBytes(run, out);
        return out;
    }

    /**
     * The byte of a {@code <0xHH>} piece, or -1: six UTF-8 bytes whose middle two Rust
     * {@code u8::from_str_radix(_, 16)} accepts (two hex digits of either case, or {@code +}
     * and one digit).
     */
    static int byteToken(String t) {
        if (t.length() != 6 || !t.startsWith("<0x") || t.charAt(5) != '>') return -1;
        int lo = hexDigit(t.charAt(4));
        if (lo < 0) return -1;
        char a = t.charAt(3);
        if (a == '+') return lo;
        int hi = hexDigit(a);
        return hi < 0 ? -1 : hi * 16 + lo;
    }

    private static int hexDigit(char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    }

    private static void flushBytes(ByteArrayOutputStream run, List<String> out) {
        if (run.size() == 0) return;
        byte[] bytes = run.toByteArray();
        run.reset();
        try {
            out.add(StandardCharsets.UTF_8.newDecoder()
                    .onMalformedInput(CodingErrorAction.REPORT)
                    .onUnmappableCharacter(CodingErrorAction.REPORT)
                    .decode(ByteBuffer.wrap(bytes))
                    .toString());
        } catch (CharacterCodingException e) {
            for (int i = 0; i < bytes.length; i++) out.add(String.valueOf(REPLACEMENT_CHAR));
        }
    }

    // ---------------------------------------------------------------------------------------------
    // JSON
    // ---------------------------------------------------------------------------------------------

    /** The tokenizer.json {@code decoder} object of this chain. */
    public Map<String, Object> toJson() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("type", kind.jsonType);
        switch (kind) {
            case BYTE_LEVEL:
                m.put("add_prefix_space", params.addPrefixSpace);
                m.put("trim_offsets", params.trimOffsets);
                m.put("use_regex", params.useRegex);
                break;
            case METASPACE:
                m.put("replacement", new String(Character.toChars(params.replacement)));
                m.put("prepend_scheme", params.prependScheme.name().toLowerCase(Locale.ROOT));
                m.put("split", params.split);
                break;
            case WORD_PIECE:
                m.put("prefix", params.prefix);
                m.put("cleanup", params.cleanup);
                break;
            case BPE:
                m.put("suffix", params.content);
                break;
            case REPLACE:
                m.put("pattern", Collections.singletonMap(params.regex ? "Regex" : "String", params.pattern));
                m.put("content", params.content);
                break;
            case STRIP:
                m.put("content", new String(Character.toChars(params.stripChar)));
                m.put("start", (long) params.start);
                m.put("stop", (long) params.stop);
                break;
            case SEQUENCE:
                List<Object> list = new ArrayList<>(children.size());
                for (TokenDecoder d : children) list.add(d.toJson());
                m.put("decoders", list);
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
     * Reads {@link #toCanonicalJson()} output, or any tokenizer.json {@code decoder} object as
     * text.
     *
     * @throws GgufFormatException when the text is not a valid decoder, or is a CTC decoder
     */
    public static TokenDecoder fromCanonicalJson(String json) {
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
     * Reads a parsed tokenizer.json {@code decoder} object; {@code source} names it in messages.
     * JSON {@code null} is the identity. Members Hugging Face defaults may be absent; unknown
     * types and members are rejected.
     *
     * @throws GgufFormatException when the value is not a valid decoder, or is a CTC decoder
     */
    public static TokenDecoder fromJson(Object parsed, String source) {
        String name = source != null ? source : DEFAULT_SOURCE;
        if (parsed == null) return IDENTITY;
        try {
            return read(JsonView.of(parsed, name));
        } catch (JsonException | IllegalArgumentException e) {
            if (e instanceof GgufFormatException) throw (GgufFormatException) e;
            throw new GgufFormatException("Invalid " + name + ": " + e.getMessage(), e);
        }
    }

    private static TokenDecoder read(JsonView v) {
        Kind kind = Kind.ofJsonType(v.str("type"), v);
        switch (kind) {
            case BYTE_LEVEL:
                Normalizer.allowKeys(v, "add_prefix_space", "trim_offsets", "use_regex");
                return byteLevel(v.optBool("add_prefix_space", true), v.optBool("trim_offsets", true),
                        v.optBool("use_regex", true));
            case METASPACE:
                Normalizer.allowKeys(v, "replacement", "prepend_scheme", "split", "add_prefix_space", "str_rep");
                return metaspace(singleCodePoint(v, "replacement"), readPrependScheme(v), v.optBool("split", true));
            case WORD_PIECE:
                Normalizer.allowKeys(v, "prefix", "cleanup");
                return wordPiece(v.optStr("prefix", "##"), v.optBool("cleanup", true));
            case BPE:
                Normalizer.allowKeys(v, "suffix");
                return bpe(v.optStr("suffix", "</w>"));
            case REPLACE: {
                Normalizer.allowKeys(v, "pattern", "content");
                JsonView pattern = v.obj("pattern");
                boolean regex = Normalizer.readPatternKind(pattern);
                return replace(pattern.str(regex ? "Regex" : "String"), regex, v.str("content"));
            }
            case STRIP:
                Normalizer.allowKeys(v, "content", "start", "stop");
                return strip(singleCodePoint(v, "content"), nonNegative(v, "start"), nonNegative(v, "stop"));
            case SEQUENCE: {
                Normalizer.allowKeys(v, "decoders");
                List<Object> raw = v.list("decoders");
                List<TokenDecoder> list = new ArrayList<>(raw.size());
                String base = v.path().isEmpty() ? "decoders" : v.path() + ".decoders";
                for (int i = 0; i < raw.size(); i++) {
                    list.add(read(JsonView.of(raw.get(i), v.source(), base + "[" + i + "]")));
                }
                return sequence(list);
            }
            case BYTE_FALLBACK:
                Normalizer.allowKeys(v);
                return byteFallback();
            case FUSE:
                Normalizer.allowKeys(v);
                return fuse();
            default:
                throw new IllegalStateException("unhandled decoder kind " + kind);
        }
    }

    /**
     * The prepend scheme of a {@code Metaspace} pre-tokenizer or decoder, as tokenizers deserializes
     * it: {@code prepend_scheme} ({@code always}, {@code first} or {@code never}; {@code always}
     * when absent), and {@code never} whenever the legacy {@code add_prefix_space} is false, which
     * earlier tokenizers releases apply over any scheme and current ones accept only beside
     * {@code prepend_scheme: never}.
     */
    public static PreTokenizerSpec.PrependScheme readPrependScheme(JsonView v) {
        PreTokenizerSpec.PrependScheme scheme = PreTokenizerSpec.PrependScheme.ALWAYS;
        if (v.raw("prepend_scheme") != null) {
            String s = v.str("prepend_scheme");
            switch (s) {
                case "always": scheme = PreTokenizerSpec.PrependScheme.ALWAYS; break;
                case "first": scheme = PreTokenizerSpec.PrependScheme.FIRST; break;
                case "never": scheme = PreTokenizerSpec.PrependScheme.NEVER; break;
                default:
                    throw new JsonException(v.where("prepend_scheme") + ": unknown value '" + s
                            + "', expected always, first or never");
            }
        }
        if (v.raw("add_prefix_space") != null && !v.bool("add_prefix_space")) {
            scheme = PreTokenizerSpec.PrependScheme.NEVER;
        }
        return scheme;
    }

    private static int singleCodePoint(JsonView v, String key) {
        String s = v.str(key);
        if (s.isEmpty() || s.codePointCount(0, s.length()) != 1) {
            throw new JsonException(v.where(key) + ": expected exactly one character, got '" + s + "'");
        }
        return s.codePointAt(0);
    }

    private static int nonNegative(JsonView v, String key) {
        int n = v.i32(key);
        if (n < 0) throw new JsonException(v.where(key) + ": expected a non-negative count, got " + n);
        return n;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof TokenDecoder)) return false;
        TokenDecoder d = (TokenDecoder) o;
        return kind == d.kind && params.equals(d.params) && children.equals(d.children);
    }

    @Override
    public int hashCode() {
        return Objects.hash(kind, params, children);
    }

    @Override
    public String toString() {
        return "TokenDecoder" + toJson();
    }
}

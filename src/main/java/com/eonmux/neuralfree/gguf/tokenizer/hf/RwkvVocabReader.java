package com.eonmux.neuralfree.gguf.tokenizer.hf;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * An RWKV World vocabulary ({@code rwkv_vocab_v20230424.txt}) in the format llama.cpp's
 * {@code _set_vocab_rwkv_world} reads (LC/conversion/base.py:2420-2463): one line per token,
 * {@code <id> <Python str or bytes literal> <byte length>}, the literal possibly holding spaces.
 *
 * <p>A str literal is encoded as UTF-8; the encoded length must equal the length field. Ids must run 1, 2, 3, ...
 * in line order: the converter numbers tokens by line after the implicit {@code <s>} at id 0, and the RWKV tokenizer
 * numbers them by the id field, so only that layout means the same vocabulary to both. {@link #reprText(int)} is the
 * token text the converter stores ({@code repr(token)[2:-1]}). Instances are immutable.</p>
 */
public final class RwkvVocabReader {

    private final byte[][] tokens;

    private RwkvVocabReader(byte[][] tokens) {
        this.tokens = tokens;
    }

    /**
     * Reads {@code file} (strict UTF-8, Python text-mode lines, at most {@link HfTokenizerFiles#MAX_FILE_BYTES}).
     *
     * @throws IOException {@code "Invalid Hugging Face checkpoint '<dir>': <file>: line <n>: ..."} for a bad line
     */
    public static RwkvVocabReader read(Path file) throws IOException {
        Objects.requireNonNull(file, "file");
        String text = HfTokenizerFiles.readText(file);
        return parse(text, HfTokenizerFiles.filePrefix(file));
    }

    /**
     * Parses the text of a vocabulary file.
     *
     * @throws IOException {@code "Invalid RWKV vocabulary: line <n>: ..."} for a bad line
     */
    public static RwkvVocabReader parse(String text) throws IOException {
        Objects.requireNonNull(text, "text");
        return parse(text, "Invalid RWKV vocabulary: ");
    }

    private static RwkvVocabReader parse(String text, String prefix) throws IOException {
        List<String> lines = PythonText.lines(text);
        if (lines.size() >= HfTokenizerFiles.MAX_TOKENS) {
            throw new IOException(prefix + lines.size() + " lines exceed the limit of "
                    + (HfTokenizerFiles.MAX_TOKENS - 1) + " tokens");
        }
        List<byte[]> tokens = new ArrayList<>(lines.size());
        for (int i = 0; i < lines.size(); i++) {
            String where = prefix + "line " + (i + 1) + ": ";
            String line = lines.get(i);
            int first = line.indexOf(' ');
            int last = line.lastIndexOf(' ');
            if (first < 0 || first == last) {
                throw new IOException(where + "expected '<id> <literal> <length>'");
            }
            long id = parseInt(line.substring(0, first), where, "id");
            if (id != i + 1) {
                throw new IOException(where + "id " + id + " is not " + (i + 1) + "; ids must run 1, 2, 3, ...");
            }
            long length = parseInt(line.substring(last + 1), where, "length");
            byte[] token = new PythonLiteral(line.substring(first + 1, last), where).parse();
            if (token.length != length) {
                throw new IOException(where + "the token has " + token.length + " bytes, the length field says "
                        + length);
            }
            tokens.add(token);
        }
        return new RwkvVocabReader(tokens.toArray(new byte[0][]));
    }

    /** Number of tokens; token {@code i} has id {@code i + 1}. */
    public int size() { return tokens.length; }

    /** The bytes of token {@code i} (a copy). */
    public byte[] bytes(int i) { return tokens[i].clone(); }

    /** The token text the converter stores for token {@code i}: {@code repr(token)[2:-1]}. */
    public String reprText(int i) { return pythonBytesRepr(tokens[i]); }

    @Override
    public String toString() {
        return "RwkvVocabReader{" + tokens.length + " tokens}";
    }

    /**
     * The body of Python's {@code repr(bytes)} without the {@code b} and the quotes: the quote is {@code "} when the
     * bytes hold {@code '} but no {@code "}, else {@code '}; backslash and the quote are escaped; tab, newline and
     * carriage return use their escapes; other bytes below 0x20 or from 0x7F are {@code \xhh}.
     */
    public static String pythonBytesRepr(byte[] b) {
        boolean hasSingle = false;
        boolean hasDouble = false;
        for (byte x : b) {
            hasSingle |= x == '\'';
            hasDouble |= x == '"';
        }
        char quote = hasSingle && !hasDouble ? '"' : '\'';
        StringBuilder sb = new StringBuilder(b.length);
        for (byte x : b) {
            int c = x & 0xFF;
            if (c == quote || c == '\\') {
                sb.append('\\').append((char) c);
            } else if (c == '\t') {
                sb.append("\\t");
            } else if (c == '\n') {
                sb.append("\\n");
            } else if (c == '\r') {
                sb.append("\\r");
            } else if (c < 0x20 || c >= 0x7F) {
                sb.append(String.format(Locale.ROOT, "\\x%02x", c));
            } else {
                sb.append((char) c);
            }
        }
        return sb.toString();
    }

    private static long parseInt(String s, String where, String what) throws IOException {
        if (s.isEmpty() || s.length() > 10) {
            throw new IOException(where + what + " '" + s + "' is not a non-negative integer");
        }
        long v = 0;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c < '0' || c > '9') {
                throw new IOException(where + what + " '" + s + "' is not a non-negative integer");
            }
            v = v * 10 + (c - '0');
        }
        return v;
    }

    /**
     * One Python string or bytes literal as {@code ast.literal_eval} reads it: prefixes {@code b}, {@code u},
     * {@code r} and their combinations, single or triple quotes, and the escapes of the Python language reference.
     * Its value as bytes: a bytes literal directly, a str literal encoded as UTF-8 (lone surrogates are an error).
     */
    private static final class PythonLiteral {
        private final String s;
        private final String where;
        private int pos;

        PythonLiteral(String s, String where) {
            this.s = s;
            this.where = where;
        }

        byte[] parse() throws IOException {
            while (pos < s.length() && (s.charAt(pos) == ' ' || s.charAt(pos) == '\t')) {
                pos++;
            }
            boolean bytes = false;
            boolean raw = false;
            while (pos < s.length() && "bBuUrR".indexOf(s.charAt(pos)) >= 0) {
                char p = Character.toLowerCase(s.charAt(pos++));
                bytes |= p == 'b';
                raw |= p == 'r';
            }
            if (pos >= s.length() || (s.charAt(pos) != '\'' && s.charAt(pos) != '"')) {
                throw error("expected a Python string or bytes literal, got '" + s + "'");
            }
            char q = s.charAt(pos);
            String quote = s.startsWith("" + q + q + q, pos) ? "" + q + q + q : "" + q;
            pos += quote.length();
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            while (true) {
                if (pos >= s.length()) {
                    throw error("unterminated literal '" + s + "'");
                }
                if (s.startsWith(quote, pos)) {
                    pos += quote.length();
                    break;
                }
                int cp = s.codePointAt(pos);
                if (cp == '\\') {
                    escape(out, bytes, raw);
                } else {
                    appendSource(out, bytes);
                }
            }
            while (pos < s.length() && Character.isWhitespace(s.charAt(pos))) {
                pos++;
            }
            if (pos != s.length()) {
                throw error("unexpected text after the literal: '" + s.substring(pos) + "'");
            }
            return out.toByteArray();
        }

        private void escape(ByteArrayOutputStream out, boolean bytes, boolean raw) throws IOException {
            if (pos + 1 >= s.length()) {
                throw error("literal ends with a backslash");
            }
            char c = s.charAt(pos + 1);
            if (raw) {
                out.write('\\');
                pos++;
                appendSource(out, bytes);
                return;
            }
            pos += 2;
            switch (c) {
                case '\\': out.write('\\'); return;
                case '\'': out.write('\''); return;
                case '"': out.write('"'); return;
                case 'a': out.write(7); return;
                case 'b': out.write(8); return;
                case 'f': out.write(12); return;
                case 'n': out.write('\n'); return;
                case 'r': out.write('\r'); return;
                case 't': out.write('\t'); return;
                case 'v': out.write(11); return;
                case 'x': append(out, hex(2), bytes); return;
                default: break;
            }
            if (c >= '0' && c <= '7') {
                int v = c - '0';
                for (int k = 0; k < 2 && pos < s.length() && s.charAt(pos) >= '0' && s.charAt(pos) <= '7'; k++) {
                    v = v * 8 + (s.charAt(pos++) - '0');
                }
                if (bytes && v > 0xFF) {
                    throw error("octal escape \\" + Integer.toOctalString(v) + " exceeds a byte");
                }
                append(out, v, bytes);
                return;
            }
            if (!bytes && (c == 'u' || c == 'U')) {
                int cp = hex(c == 'u' ? 4 : 8);
                if (cp > Character.MAX_CODE_POINT) {
                    throw error("escape \\U" + Integer.toHexString(cp) + " is not a code point");
                }
                append(out, cp, false);
                return;
            }
            if (!bytes && c == 'N') {
                append(out, namedCharacter(), false);
                return;
            }
            out.write('\\');
            pos -= 1;
            appendSource(out, bytes);
        }

        private int hex(int digits) throws IOException {
            if (pos + digits > s.length()) {
                throw error("truncated escape in '" + s + "'");
            }
            int v = 0;
            for (int k = 0; k < digits; k++) {
                int d = Character.digit(s.charAt(pos + k), 16);
                if (d < 0 || s.charAt(pos + k) > 'f') {
                    throw error("malformed hexadecimal escape in '" + s + "'");
                }
                v = v * 16 + d;
            }
            pos += digits;
            return v;
        }

        private int namedCharacter() throws IOException {
            int close = s.indexOf('}', pos);
            if (pos >= s.length() || s.charAt(pos) != '{' || close < 0) {
                throw error("malformed \\N{...} escape in '" + s + "'");
            }
            String name = s.substring(pos + 1, close);
            pos = close + 1;
            try {
                return Character.codePointOf(name);
            } catch (IllegalArgumentException e) {
                throw error("unknown character name '" + name + "'");
            }
        }

        /** Appends the source character at {@code pos} and moves past it; bytes literals hold only ASCII. */
        private void appendSource(ByteArrayOutputStream out, boolean bytes) throws IOException {
            int cp = s.codePointAt(pos);
            if (bytes && cp >= 0x80) {
                throw error("bytes literal holds the non-ASCII character U+" + String.format(Locale.ROOT, "%04X", cp));
            }
            append(out, cp, bytes);
            pos += Character.charCount(cp);
        }

        /** Appends one value: a byte for bytes literals, the UTF-8 encoding of a code point for str literals. */
        private void append(ByteArrayOutputStream out, int value, boolean bytes) throws IOException {
            if (bytes) {
                out.write(value);
                return;
            }
            if (value >= Character.MIN_SURROGATE && value <= Character.MAX_SURROGATE) {
                throw error("str literal holds the surrogate U+" + String.format(Locale.ROOT, "%04X", value)
                        + ", which UTF-8 cannot encode");
            }
            byte[] utf8 = new String(Character.toChars(value)).getBytes(StandardCharsets.UTF_8);
            out.write(utf8, 0, utf8.length);
        }

        private IOException error(String detail) {
            return new IOException(where + detail);
        }
    }
}

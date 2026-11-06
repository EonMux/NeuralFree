package com.eonmux.neuralfree.gguf.chat.tools;

import com.eonmux.neuralfree.utils.json.Json;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * The Gemma 4 format: a call is the function name and {@code key:value} arguments between two special tokens:
 *
 * <pre>{@code
 * <|tool_call>call:get_weather{location:<|"|>Paris<|"|>,days:3}<tool_call|>
 * }</pre>
 *
 * <p>A value is a string between {@code <|"|>} delimiters (its raw text), or a nested {@code {…}} object, a
 * {@code […]} array, or a bare number, boolean or null written as JSON (llama.cpp's {@code gemma4} grammar). Keys
 * and values separate on commas and colons outside strings and containers; a bare top-level value is typed by the
 * parameter schema. While a call streams, a value is reported once it closes — an open string's raw text included,
 * an open bare value held back — so every partial argument text is a prefix of the final one.
 */
final class Gemma4ToolParser extends BlockToolParser {

    private static final String OPEN = "<|tool_call>";
    private static final String CLOSE = "<tool_call|>";
    private static final String STRING = "<|\"|>";

    /** The members loop found no value where the text ended. */
    private static final int INVALID = -2;

    /** The text ends inside a bare value: what was read before it stands. */
    private static final int ENDED = -3;

    Gemma4ToolParser() {
        super(Arrays.asList(OPEN));
    }

    @Override
    protected Block readBlock(String text, int at, String opener, boolean partial) {
        int i = at + OPEN.length();
        if (!text.startsWith("call:", i)) {
            return partial && prefixOf(text, i, "call:") ? Block.incomplete(new ArrayList<>(), at)
                    : Block.malformed(i + 1);
        }
        i += "call:".length();
        int brace = text.indexOf('{', i);
        if (brace < 0) {
            return partial ? Block.incomplete(new ArrayList<>(), at) : Block.malformed(Math.max(i + 1, at + OPEN.length()));
        }
        String name = text.substring(i, brace).trim();
        if (name.isEmpty()) {
            return Block.malformed(brace);
        }
        StringBuilder members = new StringBuilder();
        int end = members(text, brace + 1, members, partial);
        if (end == INVALID) {
            return partial ? Block.incomplete(new ArrayList<>(), at)
                    : Block.malformed(Math.max(brace + 1, at + OPEN.length()));
        }
        boolean closed = end >= 0;
        List<ToolCall> calls = new ArrayList<>();
        calls.add(new ToolCall(name, closed ? "{" + members + "}" : "{" + members, null, closed));
        if (!closed) {
            return partial ? Block.incomplete(calls, at) : Block.complete(calls, text.length());
        }
        int k = JsonScan.skipWs(text, end);
        if (text.startsWith(CLOSE, k)) {
            return Block.complete(calls, k + CLOSE.length());
        }
        if (partial && (k >= text.length() || prefixOf(text, k, CLOSE))) {
            return Block.incomplete(calls, at);
        }
        return Block.complete(calls, k >= text.length() ? text.length() : k);
    }

    /**
     * The members of the arguments object opening after its brace, appended as JSON text to {@code members}.
     * Returns where the closing brace ends, -1 when the text ends inside the object, {@link #INVALID} when the
     * arguments are none.
     */
    private int members(String text, int from, StringBuilder members, boolean partial) {
        int i = from;
        while (true) {
            i = skipSeparators(text, i);
            if (i >= text.length()) {
                return -1;
            }
            if (text.charAt(i) == '}') {
                return i + 1;
            }
            int colon = text.indexOf(':', i);
            int close = text.indexOf('}', i);
            if (colon < 0) {
                return partial ? ENDED : INVALID;
            }
            if (close >= 0 && close < colon) {
                return INVALID;
            }
            String key = unwrap(text.substring(i, colon).trim());
            if (key.isEmpty()) {
                return INVALID;
            }
            StringBuilder value = new StringBuilder();
            int after = value(text, colon + 1, value, partial);
            if (after == INVALID) {
                return INVALID;
            }
            if (after == ENDED) {
                // the text ends inside the bare value: what was read before it stands
                return -1;
            }
            if (members.length() > 0) {
                members.append(',');
            }
            members.append(quote(key)).append(':').append(value);
            if (after < 0) {
                return -1;
            }
            i = after;
        }
    }

    /**
     * The value at {@code from}, appended as JSON text to {@code out}: a {@code <|"|>} string's raw text, a
     * nested container, or a bare value read as JSON. Returns where the value ends, -1 when the text ends inside a
     * top-level string, {@link #ENDED} when the text ends inside a bare value, {@link #INVALID} when the value is
     * none.
     */
    private int value(String text, int from, StringBuilder out, boolean partial) {
        int i = skipSeparators(text, from);
        if (i >= text.length()) {
            return ENDED;
        }
        if (text.startsWith(STRING, i)) {
            int vs = i + STRING.length();
            int ve = text.indexOf(STRING, vs);
            if (ve < 0) {
                String raw = text.substring(vs);
                if (partial) {
                    raw = stripPartialDelim(raw);
                }
                out.append(quoteOpen(raw));
                return -1;
            }
            out.append(quote(text.substring(vs, ve)));
            return ve + STRING.length();
        }
        char c = text.charAt(i);
        if (c == '{' || c == '[') {
            int[] span = containerSpan(text, i);
            if (span == null) {
                return INVALID;
            }
            String json = containerJson(text.substring(i + 1, span[0]), c == '{');
            if (json == null) {
                return INVALID;
            }
            out.append(json);
            return span[1];
        }
        int end = i;
        while (end < text.length() && ",}]".indexOf(text.charAt(end)) < 0) {
            end++;
        }
        String raw = text.substring(i, end).trim();
        if (end >= text.length() && partial) {
            return ENDED;
        }
        Object parsed = JsonScan.parseOrNull(raw);
        out.append(parsed == null ? quote(raw) : json(parsed));
        return end;
    }

    /** The JSON text of the members written in {@code inner} between the braces or brackets of one container. */
    private static String containerJson(String inner, boolean object) {
        StringBuilder out = new StringBuilder(object ? "{" : "[");
        int i = 0;
        while (true) {
            i = skipSeparators(inner, i);
            if (i >= inner.length()) {
                break;
            }
            if (object) {
                int colon = inner.indexOf(':', i);
                if (colon < 0) {
                    return null;
                }
                String key = unwrap(inner.substring(i, colon).trim());
                if (key.isEmpty()) {
                    return null;
                }
                if (out.length() > 1) {
                    out.append(',');
                }
                out.append(quote(key)).append(':');
                i = colon + 1;
                i = skipSeparators(inner, i);
                if (i >= inner.length()) {
                    return null;
                }
            } else if (out.length() > 1) {
                out.append(',');
            }
            if (inner.startsWith(STRING, i)) {
                int ve = inner.indexOf(STRING, i + STRING.length());
                if (ve < 0) {
                    return null;
                }
                out.append(quote(inner.substring(i + STRING.length(), ve)));
                i = ve + STRING.length();
                continue;
            }
            char c = inner.charAt(i);
            if (c == '{' || c == '[') {
                int[] span = containerSpan(inner, i);
                if (span == null) {
                    return null;
                }
                String json = containerJson(inner.substring(i + 1, span[0]), c == '{');
                if (json == null) {
                    return null;
                }
                out.append(json);
                i = span[1];
                continue;
            }
            int end = i;
            while (end < inner.length() && ",}]".indexOf(inner.charAt(end)) < 0) {
                end++;
            }
            String raw = inner.substring(i, end).trim();
            Object parsed = JsonScan.parseOrNull(raw);
            out.append(parsed == null ? quote(raw) : json(parsed));
            i = end;
        }
        return out.append(object ? "}" : "]").toString();
    }

    /** Where the container opening at {@code from} ends: the index of its close and one past it, or null. */
    private static int[] containerSpan(String text, int from) {
        char open = text.charAt(from);
        char close = open == '{' ? '}' : ']';
        int depth = 0;
        int i = from;
        while (i < text.length()) {
            if (text.startsWith(STRING, i)) {
                int ve = text.indexOf(STRING, i + STRING.length());
                i = ve < 0 ? text.length() : ve + STRING.length();
                continue;
            }
            char c = text.charAt(i);
            if (c == open) {
                depth++;
            } else if (c == close) {
                depth--;
                if (depth == 0) {
                    return new int[]{i, i + 1};
                }
            }
            i++;
        }
        return null;
    }

    private static int skipSeparators(String text, int i) {
        while (i < text.length()) {
            char c = text.charAt(i);
            if (c == ' ' || c == ',' || c == '\n' || c == '\t' || c == '\r') {
                i++;
            } else {
                break;
            }
        }
        return i;
    }

    /** The text without one wrapping {@code <|"|>} pair, or the text itself. */
    private static String unwrap(String raw) {
        if (raw.startsWith(STRING) && raw.endsWith(STRING) && raw.length() >= 2 * STRING.length()) {
            return raw.substring(STRING.length(), raw.length() - STRING.length());
        }
        return raw;
    }

    /** The value as JSON text: a string in quotes, anything else as its JSON. */
    private static String json(Object value) {
        if (value instanceof String) {
            return quote((String) value);
        }
        StringBuilder out = new StringBuilder();
        try {
            Json.write(value, Json.Style.API, out);
        } catch (IOException e) {
            throw new IllegalStateException("a typed argument does not write back to JSON", e);
        }
        return out.toString();
    }

    private static String quote(String s) {
        return quoteOpen(s) + '"';
    }

    /** The string so far as JSON text: the opening quote and the escaped body, no closing quote. */
    private static String quoteOpen(String s) {
        StringBuilder out = new StringBuilder("\"");
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '"' || c == '\\') {
                out.append('\\');
            }
            out.append(c);
        }
        return out.toString();
    }

    /** The text without a trailing proper prefix of the string delimiter, which may still be arriving. */
    private static String stripPartialDelim(String raw) {
        for (int k = STRING.length() - 1; k > 0; k--) {
            if (raw.endsWith(STRING.substring(0, k))) {
                return raw.substring(0, raw.length() - k);
            }
        }
        return raw;
    }
}

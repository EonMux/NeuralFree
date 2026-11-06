package com.eonmux.neuralfree.gguf.chat.tools;

import com.eonmux.neuralfree.utils.json.Json;

import java.util.ArrayList;
import java.util.List;

/**
 * Structural scanning of JSON text that may be cut off: where a value ends, and the member spans of an object read
 * so far. Model output is scanned before it is parsed, so the parsers can tell a value the next tokens will finish
 * ({@link #INCOMPLETE}) from text that is no JSON value ({@link #INVALID}). Complete values are then parsed with
 * {@link Json#parse} under {@link Json.Limits#UNTRUSTED}, which validates what the scan does not (commas, colons,
 * literals, escapes).
 */
final class JsonScan {

    /** The value runs past the end of the text. */
    static final int INCOMPLETE = -1;
    /** The text at the position is no JSON value. */
    static final int INVALID = -2;

    private JsonScan() {
    }

    /** The first index at or after {@code i} that is not JSON whitespace. */
    static int skipWs(CharSequence s, int i) {
        while (i < s.length()) {
            char c = s.charAt(i);
            if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
                break;
            }
            i++;
        }
        return i;
    }

    /**
     * The index after the JSON value that starts at {@code i} (no leading whitespace), {@link #INCOMPLETE} when the
     * text ends inside it, or {@link #INVALID} when no value starts there. A number or literal at the very end of
     * the text is incomplete: more digits or letters could follow.
     */
    static int valueEnd(CharSequence s, int i) {
        if (i >= s.length()) {
            return INCOMPLETE;
        }
        char c = s.charAt(i);
        if (c == '"') {
            return stringEnd(s, i);
        }
        if (c == '{' || c == '[') {
            return containerEnd(s, i);
        }
        if (c == '-' || (c >= '0' && c <= '9')) {
            int j = i + 1;
            while (j < s.length() && "0123456789+-.eE".indexOf(s.charAt(j)) >= 0) {
                j++;
            }
            return j == s.length() ? INCOMPLETE : j;
        }
        for (String lit : new String[]{"true", "false", "null"}) {
            int n = Math.min(lit.length(), s.length() - i);
            if (lit.regionMatches(0, s.toString(), i, n)) {
                return n < lit.length() ? INCOMPLETE : i + lit.length();
            }
        }
        return INVALID;
    }

    /** The index after the string literal starting at the quote {@code i}, or {@link #INCOMPLETE}. */
    static int stringEnd(CharSequence s, int i) {
        int j = i + 1;
        while (j < s.length()) {
            char c = s.charAt(j);
            if (c == '\\') {
                j += 2;
                continue;
            }
            if (c == '"') {
                return j + 1;
            }
            j++;
        }
        return INCOMPLETE;
    }

    private static int containerEnd(CharSequence s, int i) {
        int depth = 0;
        int j = i;
        while (j < s.length()) {
            char c = s.charAt(j);
            if (c == '"') {
                int e = stringEnd(s, j);
                if (e < 0) {
                    return INCOMPLETE;
                }
                j = e;
                continue;
            }
            if (c == '{' || c == '[') {
                depth++;
            } else if (c == '}' || c == ']') {
                depth--;
                if (depth == 0) {
                    return j + 1;
                }
            }
            j++;
        }
        return INCOMPLETE;
    }

    /** {@code text} parsed as one JSON value, or null when it is not valid JSON. */
    static Object parseOrNull(String text) {
        try {
            return Json.parse(text, Json.Limits.UNTRUSTED);
        } catch (RuntimeException e) {
            return null;
        }
    }

    /** One member of an object: its key, and the span of its value ({@code valueEnd} &lt; 0 when cut off). */
    static final class Member {
        final String key;
        final int valueStart;
        final int valueEnd;

        Member(String key, int valueStart, int valueEnd) {
            this.key = key;
            this.valueStart = valueStart;
            this.valueEnd = valueEnd;
        }

        boolean complete() {
            return valueEnd >= 0;
        }
    }

    /**
     * The members of the object starting at the brace {@code i}, in order, as far as the text reaches. The last
     * member may be cut off inside its value ({@link Member#complete()} false). Returns null when the text is not
     * an object's members (a non-string key, a missing colon).
     */
    static List<Member> members(CharSequence s, int i) {
        List<Member> out = new ArrayList<>();
        int j = skipWs(s, i + 1);
        if (j < s.length() && s.charAt(j) == '}') {
            return out;
        }
        while (j < s.length()) {
            if (s.charAt(j) != '"') {
                return null;
            }
            int keyEnd = stringEnd(s, j);
            if (keyEnd < 0) {
                return out;
            }
            Object key = parseOrNull(s.subSequence(j, keyEnd).toString());
            if (!(key instanceof String)) {
                return null;
            }
            j = skipWs(s, keyEnd);
            if (j >= s.length()) {
                return out;
            }
            if (s.charAt(j) != ':') {
                return null;
            }
            int vs = skipWs(s, j + 1);
            if (vs >= s.length()) {
                out.add(new Member((String) key, vs, INCOMPLETE));
                return out;
            }
            int ve = valueEnd(s, vs);
            if (ve == INVALID) {
                return null;
            }
            out.add(new Member((String) key, vs, ve));
            if (ve < 0) {
                return out;
            }
            j = skipWs(s, ve);
            if (j >= s.length()) {
                return out;
            }
            char c = s.charAt(j);
            if (c == '}') {
                return out;
            }
            if (c != ',') {
                return null;
            }
            j = skipWs(s, j + 1);
        }
        return out;
    }
}

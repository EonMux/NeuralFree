package com.eonmux.neuralfree.server;

import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * The text forms of the decision families that read free text (tools/server/server-decision.cpp):
 * {@code server_decision_escape_special_tokens} quotes {@code <|name|>} so the request's text is not parsed as
 * special tokens, and the h2o-lightning render reproduces Python's {@code json.dumps}, {@code str.lstrip} and
 * {@code str.rstrip} ({@code decision_py_*}) byte for byte, with repr-style float digits, the exponent forms of
 * {@code repr} and the {@code ensure_ascii} escapes. Immutable; every method allocates its result.
 */
final class DecisionText {

    /** The nesting bound of the kev flattening (llama.cpp's {@code RECURSIVE_LIMIT}). */
    private static final int KEV_RECURSIVE_LIMIT = 128;

    private DecisionText() {
    }

    /**
     * The kev text of a JSON value: {@link #kevRender(Object, int, int) flattened}, then
     * {@link #escapeSpecialTokens} so the request's text is not parsed as special tokens — llama.cpp's
     * {@code decision_kev_text}.
     */
    static String kevText(Object value) {
        return escapeSpecialTokens(kevRender(value, 0, 0));
    }

    /**
     * kev flattens a JSON value into text, the keys of an object kept as labels (kev/api.py's {@code render},
     * llama.cpp's {@code decision_kev_render}): a null renders empty, a boolean as Python's {@code True} or
     * {@code False}, a string as itself, an array as one dash-prefixed line per item at the next indent, an object
     * as one {@code key:} line per entry — nested values on their own indented lines, scalars after the colon —
     * and every other value as its JSON dump.
     *
     * @throws IllegalArgumentException when the value nests more than 128 levels deep or is not JSON
     */
    static String kevRender(Object value, int indent, int depth) {
        if (depth >= KEV_RECURSIVE_LIMIT) {
            throw new IllegalArgumentException("JSON is nested more than " + KEV_RECURSIVE_LIMIT + " levels deep");
        }
        String pad = " ".repeat(2 * indent);
        if (value == null) {
            return "";
        }
        if (value instanceof String s) {
            return s;
        }
        if (value instanceof Boolean b) {
            return b ? "True" : "False";
        }
        if (value instanceof List<?> list) {
            StringBuilder out = new StringBuilder();
            for (Object item : list) {
                String text = kevRender(item, indent + 1, depth + 1);
                if (out.length() > 0) {
                    out.append('\n');
                }
                out.append(pad).append("- ").append(kevStripLeading(text));
            }
            return out.toString();
        }
        if (value instanceof Map<?, ?> map) {
            StringBuilder out = new StringBuilder();
            for (Map.Entry<?, ?> entry : map.entrySet()) {
                boolean nested = entry.getValue() instanceof Map || entry.getValue() instanceof List;
                String text = kevRender(entry.getValue(), nested ? indent + 1 : 0, depth + 1);
                if (out.length() > 0) {
                    out.append('\n');
                }
                out.append(pad).append(entry.getKey()).append(nested ? ":\n" : ": ").append(text);
            }
            return out.toString();
        }
        return dumps(value, -1, ",", ":", false);
    }

    /** The leading spaces, tabs and newlines of {@code text} removed (llama.cpp's {@code find_first_not_of}). */
    private static String kevStripLeading(String text) {
        int i = 0;
        while (i < text.length()) {
            char c = text.charAt(i);
            if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
                break;
            }
            i++;
        }
        return text.substring(i);
    }

    /**
     * Writes every {@code <|name|>} of {@code text} as {@code <&#x00A6;name&#x00A6;>} (U+00A6 in place of the
     * pipes), so text from the request is not parsed as special tokens — llama.cpp's {@code
     * server_decision_escape_special_tokens}, the same result as {@code regex_replace(<\\|([A-Za-z0-9_]+)\\|>)}.
     */
    static String escapeSpecialTokens(String text) {
        StringBuilder out = null;
        int nDone = 0; // the chars of text that are in out
        int i = text.indexOf("<|");
        while (i >= 0) {
            int j = i + 2;
            while (j < text.length() && isNameChar(text.charAt(j))) {
                j++;
            }
            if (j > i + 2 && text.startsWith("|>", j)) {
                if (out == null) {
                    out = new StringBuilder();
                }
                out.append(text, nDone, i).append("<\u00a6").append(text, i + 2, j).append("\u00a6>");
                nDone = j + 2;
            }
            i = text.indexOf("<|", Math.max(i + 1, nDone));
        }
        if (out == null) {
            return text;
        }
        return out.append(text, nDone, text.length()).toString();
    }

    private static boolean isNameChar(char c) {
        return c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z' || c >= '0' && c <= '9' || c == '_';
    }

    /**
     * Python's {@code json.dumps} of a JSON value: {@code indent} below zero keeps the value on one line, keys
     * keep their order, floats use {@code repr} digits, and {@code ensureAscii} escapes every non-ASCII code
     * point (surrogate pairs above the BMP) — llama.cpp's {@code decision_py_dumps}.
     */
    static String dumps(Object value, int indent, String itemSep, String keySep, boolean ensureAscii) {
        StringBuilder out = new StringBuilder();
        dump(out, value, indent, 0, itemSep, keySep, ensureAscii);
        return out.toString();
    }

    private static void dump(StringBuilder out, Object value, int indent, int level, String itemSep,
                             String keySep, boolean ensureAscii) {
        String newline = indent < 0 ? "" : "\n" + " ".repeat(indent * level);
        switch (value) {
            case null -> out.append("null");
            case Boolean b -> out.append(b ? "true" : "false");
            case Float f -> out.append(pyFloat(f.doubleValue()));
            case Double d -> out.append(pyFloat(d));
            case Number n -> out.append(n);
            case String s -> pyStr(out, s, ensureAscii);
            case List<?> list -> {
                if (list.isEmpty()) {
                    out.append("[]");
                    return;
                }
                out.append('[').append(indent < 0 ? "" : "\n" + " ".repeat(indent * (level + 1)));
                for (int i = 0; i < list.size(); i++) {
                    if (i > 0) {
                        out.append(itemSep).append(indent < 0 ? "" : "\n" + " ".repeat(indent * (level + 1)));
                    }
                    dump(out, list.get(i), indent, level + 1, itemSep, keySep, ensureAscii);
                }
                out.append(newline).append(']');
            }
            case Map<?, ?> map -> {
                if (map.isEmpty()) {
                    out.append("{}");
                    return;
                }
                out.append('{').append(indent < 0 ? "" : "\n" + " ".repeat(indent * (level + 1)));
                boolean first = true;
                for (Map.Entry<?, ?> entry : map.entrySet()) {
                    if (!first) {
                        out.append(itemSep).append(indent < 0 ? "" : "\n" + " ".repeat(indent * (level + 1)));
                    }
                    first = false;
                    pyStr(out, String.valueOf(entry.getKey()), ensureAscii);
                    out.append(keySep);
                    dump(out, entry.getValue(), indent, level + 1, itemSep, keySep, ensureAscii);
                }
                out.append(newline).append('}');
            }
            default -> throw new IllegalArgumentException(
                    "the value is not JSON: " + value.getClass().getSimpleName());
        }
    }

    /** A string as JSON with Python's escapes: {@code ensure_ascii} escapes every code point above 0x7e. */
    private static void pyStr(StringBuilder out, String s, boolean ensureAscii) {
        out.append('"');
        for (int i = 0; i < s.length(); ) {
            int cp = s.codePointAt(i);
            switch (cp) {
                case '"' -> out.append("\\\"");
                case '\\' -> out.append("\\\\");
                case '\b' -> out.append("\\b");
                case '\f' -> out.append("\\f");
                case '\n' -> out.append("\\n");
                case '\r' -> out.append("\\r");
                case '\t' -> out.append("\\t");
                default -> {
                    if (cp < 0x20 || (ensureAscii && cp == 0x7f)) {
                        out.append("\\u").append(String.format(Locale.ROOT, "%04x", cp));
                    } else if (cp < 0x80 || !ensureAscii) {
                        out.appendCodePoint(cp);
                    } else {
                        out.append(escapeCp(cp));
                    }
                }
            }
            i += Character.charCount(cp);
        }
        out.append('"');
    }

    /** A code point as JSON escapes, a surrogate pair above the BMP (llama.cpp's {@code decision_py_escape_cp}). */
    private static String escapeCp(int cp) {
        if (cp < 0x10000) {
            return String.format(Locale.ROOT, "\\u%04x", cp);
        }
        int wide = cp - 0x10000;
        return String.format(Locale.ROOT, "\\u%04x\\u%04x", 0xd800 + (wide >> 10), 0xdc00 + (wide & 0x3ff));
    }

    /**
     * {@code repr} of a Python float: the fewest digits that read back the same, in exponent form below 1e-4 and
     * from 1e16, with a two-digit exponent sign and always a fraction (llama.cpp's {@code decision_py_float}).
     */
    private static String pyFloat(double v) {
        String sci = "";
        for (int prec = 0; prec < 17; prec++) {
            sci = String.format(Locale.ROOT, "%." + prec + "e", v);
            if (Double.parseDouble(sci) == v) {
                break;
            }
        }
        int posE = sci.indexOf('e');
        int exp = Integer.parseInt(sci.substring(posE + 1));
        String sign = sci.startsWith("-") ? "-" : "";
        StringBuilder digits = new StringBuilder();
        for (int i = 0; i < posE; i++) {
            char c = sci.charAt(i);
            if (c != '-' && c != '.') {
                digits.append(c);
            }
        }
        int n = digits.length();
        if (exp < -4 || exp >= 16) {
            String frac = n > 1 ? "." + digits.substring(1) : "";
            return sign + digits.charAt(0) + frac + String.format(Locale.ROOT, "e%c%02d", exp < 0 ? '-' : '+',
                    Math.abs(exp));
        }
        if (exp >= n - 1) {
            return sign + digits + "0".repeat(exp - (n - 1)) + ".0";
        }
        if (exp >= 0) {
            return sign + digits.substring(0, exp + 1) + "." + digits.substring(exp + 1);
        }
        return sign + "0." + "0".repeat(-exp - 1) + digits;
    }

    /** Python's {@code str.lstrip}: leading whitespace of Python's {@code str.isspace} set removed. */
    static String lstrip(String s) {
        int i = 0;
        while (i < s.length()) {
            int cp = s.codePointAt(i);
            if (!pyIsspace(cp)) {
                break;
            }
            i += Character.charCount(cp);
        }
        return s.substring(i);
    }

    /** Python's {@code str.rstrip}: trailing whitespace of Python's {@code str.isspace} set removed. */
    static String rstrip(String s) {
        int end = 0;
        for (int i = 0; i < s.length(); ) {
            int cp = s.codePointAt(i);
            i += Character.charCount(cp);
            if (!pyIsspace(cp)) {
                end = i;
            }
        }
        return s.substring(0, end);
    }

    /** Python's {@code str.isspace} of one code point (llama.cpp's {@code decision_py_isspace}). */
    private static boolean pyIsspace(int cp) {
        return cp >= 0x09 && cp <= 0x0d || cp >= 0x1c && cp <= 0x20 || cp == 0x85 || cp == 0xa0 || cp == 0x1680
                || cp >= 0x2000 && cp <= 0x200a || cp == 0x2028 || cp == 0x2029 || cp == 0x202f || cp == 0x205f
                || cp == 0x3000;
    }
}

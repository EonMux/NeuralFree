package com.eonmux.neuralfree.gguf.chat.tools;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Python literals as {@code ast.literal_eval} reads them in pythonic tool calls, converted to the JSON value model
 * ({@code LinkedHashMap}, {@code ArrayList}, {@code Long}/{@code BigInteger}/{@code Double}, {@code String},
 * {@code Boolean}, null): single- and double-quoted strings with escapes, integers and floats, {@code True},
 * {@code False}, {@code None}, lists, tuples and dicts (string keys). JSON's {@code true}/{@code false}/{@code null}
 * are accepted as well, as models mix them in.
 */
final class PyLiteral {

    private PyLiteral() {
    }

    /** A literal read: its value, where it ends (-1 cut off), or invalid. */
    static final class Result {
        final Object value;
        final int end;
        final boolean invalid;

        private Result(Object value, int end, boolean invalid) {
            this.value = value;
            this.end = end;
            this.invalid = invalid;
        }

        static Result of(Object value, int end) {
            return new Result(value, end, false);
        }

        static final Result CUT = new Result(null, -1, false);
        static final Result INVALID = new Result(null, -1, true);
    }

    static Result read(String s, int i) {
        if (i >= s.length()) {
            return Result.CUT;
        }
        char c = s.charAt(i);
        if (c == '"' || c == '\'') {
            return string(s, i, c);
        }
        if (c == '[' || c == '(') {
            return sequence(s, i, c == '[' ? ']' : ')');
        }
        if (c == '{') {
            return dict(s, i);
        }
        if (c == '-' || c == '+' || (c >= '0' && c <= '9') || c == '.') {
            return number(s, i);
        }
        String[] words = {"True", "False", "None", "true", "false", "null"};
        Object[] values = {Boolean.TRUE, Boolean.FALSE, null, Boolean.TRUE, Boolean.FALSE, null};
        for (int w = 0; w < words.length; w++) {
            String word = words[w];
            int n = Math.min(word.length(), s.length() - i);
            if (word.regionMatches(0, s, i, n)) {
                if (n < word.length()) {
                    return Result.CUT;
                }
                int e = i + word.length();
                if (e < s.length() && Character.isLetterOrDigit(s.charAt(e))) {
                    continue;
                }
                return e == s.length() ? Result.CUT : Result.of(values[w], e);
            }
        }
        return Result.INVALID;
    }

    private static Result string(String s, int i, char quote) {
        StringBuilder out = new StringBuilder();
        int j = i + 1;
        while (j < s.length()) {
            char c = s.charAt(j);
            if (c == quote) {
                return Result.of(out.toString(), j + 1);
            }
            if (c == '\\') {
                if (j + 1 >= s.length()) {
                    return Result.CUT;
                }
                char e = s.charAt(j + 1);
                switch (e) {
                    case 'n': out.append('\n'); break;
                    case 't': out.append('\t'); break;
                    case 'r': out.append('\r'); break;
                    case 'b': out.append('\b'); break;
                    case 'f': out.append('\f'); break;
                    case '0': out.append('\0'); break;
                    case 'u':
                    case 'x': {
                        int len = e == 'u' ? 4 : 2;
                        if (j + 2 + len > s.length()) {
                            return Result.CUT;
                        }
                        try {
                            out.append((char) Integer.parseInt(s.substring(j + 2, j + 2 + len), 16));
                        } catch (NumberFormatException ex) {
                            return Result.INVALID;
                        }
                        j += len;
                        break;
                    }
                    default: out.append(e); break;
                }
                j += 2;
                continue;
            }
            out.append(c);
            j++;
        }
        return Result.CUT;
    }

    private static Result number(String s, int i) {
        int j = i;
        while (j < s.length() && "0123456789+-.eE_".indexOf(s.charAt(j)) >= 0) {
            j++;
        }
        if (j == s.length()) {
            return Result.CUT;
        }
        String t = s.substring(i, j).replace("_", "");
        try {
            if (t.contains(".") || t.contains("e") || t.contains("E")) {
                return Result.of(Double.parseDouble(t), j);
            }
            BigInteger b = new BigInteger(t.startsWith("+") ? t.substring(1) : t);
            return Result.of(b.bitLength() < 64 ? (Object) b.longValue() : b, j);
        } catch (NumberFormatException e) {
            return Result.INVALID;
        }
    }

    private static Result sequence(String s, int i, char close) {
        List<Object> out = new ArrayList<>();
        int j = JsonScan.skipWs(s, i + 1);
        while (true) {
            if (j >= s.length()) {
                return Result.CUT;
            }
            if (s.charAt(j) == close) {
                return Result.of(out, j + 1);
            }
            Result v = read(s, j);
            if (v.invalid || v.end < 0) {
                return v;
            }
            out.add(v.value);
            j = JsonScan.skipWs(s, v.end);
            if (j < s.length() && s.charAt(j) == ',') {
                j = JsonScan.skipWs(s, j + 1);
            } else if (j < s.length() && s.charAt(j) != close) {
                return Result.INVALID;
            }
        }
    }

    private static Result dict(String s, int i) {
        Map<String, Object> out = new LinkedHashMap<>();
        int j = JsonScan.skipWs(s, i + 1);
        while (true) {
            if (j >= s.length()) {
                return Result.CUT;
            }
            if (s.charAt(j) == '}') {
                return Result.of(out, j + 1);
            }
            Result k = read(s, j);
            if (k.invalid || k.end < 0) {
                return k;
            }
            if (!(k.value instanceof String)) {
                return Result.INVALID;
            }
            j = JsonScan.skipWs(s, k.end);
            if (j >= s.length()) {
                return Result.CUT;
            }
            if (s.charAt(j) != ':') {
                return Result.INVALID;
            }
            Result v = read(s, JsonScan.skipWs(s, j + 1));
            if (v.invalid || v.end < 0) {
                return v;
            }
            out.put((String) k.value, v.value);
            j = JsonScan.skipWs(s, v.end);
            if (j < s.length() && s.charAt(j) == ',') {
                j = JsonScan.skipWs(s, j + 1);
            } else if (j < s.length() && s.charAt(j) != '}') {
                return Result.INVALID;
            }
        }
    }
}

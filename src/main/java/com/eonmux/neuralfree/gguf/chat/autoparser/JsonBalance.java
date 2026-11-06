package com.eonmux.neuralfree.gguf.chat.autoparser;

/**
 * Finds the end of one JSON value in a text by its delimiters, without building the value: the span a {@link Peg}
 * JSON node consumes. Strings respect backslash escapes, objects and arrays nest, and an unterminated value reports
 * {@code -1} so a partial stream can wait for more input.
 */
final class JsonBalance {

    private JsonBalance() {
    }

    /**
     * The index just past the JSON value starting at {@code at} (leading whitespace skipped), or {@code -1} when the
     * text ends inside the value or does not start one.
     */
    static int end(CharSequence text, int at) {
        int i = at;
        while (i < text.length() && Character.isWhitespace(text.charAt(i))) {
            i++;
        }
        if (i >= text.length()) {
            return -1;
        }
        char c = text.charAt(i);
        switch (c) {
            case '{':
            case '[':
                return containerEnd(text, i);
            case '"':
                return stringEnd(text, i);
            default:
                return scalarEnd(text, i);
        }
    }

    private static int containerEnd(CharSequence text, int start) {
        int depth = 0;
        boolean inString = false;
        boolean escaped = false;
        for (int i = start; i < text.length(); i++) {
            char c = text.charAt(i);
            if (inString) {
                if (escaped) {
                    escaped = false;
                } else if (c == '\\') {
                    escaped = true;
                } else if (c == '"') {
                    inString = false;
                }
                continue;
            }
            if (c == '"') {
                inString = true;
            } else if (c == '{' || c == '[') {
                depth++;
            } else if (c == '}' || c == ']') {
                depth--;
                if (depth == 0) {
                    return i + 1;
                }
            }
        }
        return -1;
    }

    private static int stringEnd(CharSequence text, int start) {
        boolean escaped = false;
        for (int i = start + 1; i < text.length(); i++) {
            char c = text.charAt(i);
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                return i + 1;
            }
        }
        return -1;
    }

    private static int scalarEnd(CharSequence text, int start) {
        int i = start;
        while (i < text.length()) {
            char c = text.charAt(i);
            if (Character.isLetterOrDigit(c) || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E') {
                i++;
            } else {
                break;
            }
        }
        if (i == start) {
            return -1;
        }
        String token = text.subSequence(start, i).toString();
        if (!isJsonScalar(token)) {
            return -1;
        }
        // A scalar that runs to the end of the text may still grow (a longer number or literal).
        return i == text.length() ? -1 : i;
    }

    private static boolean isJsonScalar(String token) {
        if (token.equals("true") || token.equals("false") || token.equals("null")) {
            return true;
        }
        int i = 0;
        if (i < token.length() && (token.charAt(i) == '-' || token.charAt(i) == '+')) {
            i++;
        }
        int digits = 0;
        while (i < token.length() && Character.isDigit(token.charAt(i))) {
            i++;
            digits++;
        }
        if (digits == 0) {
            return false;
        }
        if (i < token.length() && token.charAt(i) == '.') {
            i++;
            int fraction = 0;
            while (i < token.length() && Character.isDigit(token.charAt(i))) {
                i++;
                fraction++;
            }
            if (fraction == 0) {
                return false;
            }
        }
        if (i < token.length() && (token.charAt(i) == 'e' || token.charAt(i) == 'E')) {
            i++;
            if (i < token.length() && (token.charAt(i) == '-' || token.charAt(i) == '+')) {
                i++;
            }
            int exponent = 0;
            while (i < token.length() && Character.isDigit(token.charAt(i))) {
                i++;
                exponent++;
            }
            if (exponent == 0) {
                return false;
            }
        }
        return i == token.length();
    }
}

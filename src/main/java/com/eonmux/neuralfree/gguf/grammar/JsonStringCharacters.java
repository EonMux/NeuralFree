package com.eonmux.neuralfree.gguf.grammar;

import java.util.ArrayList;
import java.util.List;

/**
 * The GBNF of decoded string characters written inside a JSON string: what a JSON-schema {@code pattern},
 * {@code patternProperties} or {@code propertyNames} regex matches is the decoded string, while the grammar writes
 * its JSON encoding. A literal character becomes its canonical encoding ({@code "} as {@code \"}, {@code \} as
 * {@code \\}, a control character as its short or six-character <code>&#92;u00XX</code> escape, DEL as
 * <code>&#92;u007f</code>, the way llama.cpp's {@code char} rule writes no raw control or DEL); a character set
 * becomes the class of its characters a JSON string holds raw, plus the canonical encoding of each quote,
 * backslash, control character or DEL the set holds. The written text decodes to exactly the matched characters.
 * Stateless.
 */
final class JsonStringCharacters {

    private static final int QUOTE = 0x22;
    private static final int BACKSLASH = 0x5C;
    private static final int DELETE = 0x7F;

    /** The characters a JSON string never holds raw: the controls, the quote, the backslash and DEL. */
    private static final RegexCharSet UNSAFE = RegexCharSet.range(0x00, 0x1F).union(RegexCharSet.single(QUOTE))
            .union(RegexCharSet.single(BACKSLASH)).union(RegexCharSet.single(DELETE));

    private JsonStringCharacters() {
    }

    /** The GBNF literal text (without its quotes) writing {@code codePoint} JSON-encoded. */
    static String literal(int codePoint) {
        switch (codePoint) {
            case QUOTE: return "\\\\\\\"";
            case BACKSLASH: return "\\\\\\\\";
            case '\b': return "\\\\b";
            case '\f': return "\\\\f";
            case '\n': return "\\\\n";
            case '\r': return "\\\\r";
            case '\t': return "\\\\t";
            default:
                if (codePoint < 0x20 || codePoint == DELETE) {
                    return String.format("\\\\u%04x", codePoint);
                }
                return new String(Character.toChars(codePoint));
        }
    }

    /** The GBNF of any one string character, {@code [^]}. */
    static String any() {
        return set(RegexCharSet.range(0, RegexCharSet.MAX_CODE_POINT));
    }

    /** The GBNF of a non-dotAll {@code .}: any string character but a line terminator. */
    static String dot() {
        return set(RegexCharSet.NON_LINE_TERMINATOR);
    }

    /**
     * The GBNF alternatives of one character of {@code members}, or null when the set is empty. The raw characters
     * are written as the shorter of their class and its negated complement; every member a JSON string cannot hold
     * raw is one literal alternative of its encoding.
     */
    static String set(RegexCharSet members) {
        List<String> alternatives = new ArrayList<>();
        RegexCharSet raw = members.minus(UNSAFE);
        if (!raw.isEmpty()) {
            String listed = raw.toGbnf(false);
            String excluded = raw.complement().toGbnf(true);
            alternatives.add(excluded.length() <= listed.length() ? excluded : listed);
        }
        for (int codePoint = 0; codePoint <= DELETE; codePoint++) {
            if (UNSAFE.contains(codePoint) && members.contains(codePoint)) {
                alternatives.add("\"" + literal(codePoint) + "\"");
            }
        }
        if (alternatives.isEmpty()) {
            return null;
        }
        return alternatives.size() == 1 ? alternatives.get(0) : "(" + String.join(" | ", alternatives) + ")";
    }
}

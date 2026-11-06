package com.eonmux.neuralfree.gguf.grammar;

/**
 * One character element of an output regex, decoded: a code point (a plain character or a
 * character escape) or a class shorthand ({@code \d \D \w \W \s \S}), together with the number
 * of pattern characters it spans. {@link RegexToGbnf} reads literal runs and character-class
 * members through {@link #at}.
 *
 * <p>The escapes are ECMAScript's (ECMA-262 {@code CharacterEscape}) plus the Python
 * {@code re} forms a vLLM {@code guided_regex} commonly carries: {@code \t \n \r \f \v};
 * {@code \0} (not followed by a digit); {@code \xHH}; {@code \}{@code uHHHH} (a surrogate pair
 * of two such escapes is one code point); {@code \}{@code u{H…}}; {@code \UHHHHHHHH};
 * {@code \cX}; {@code \b} as BACKSPACE inside a class; and the identity escape of any
 * character that is not a letter or digit ({@code \. \/ \- \# …}). Assertions and references
 * with no grammar equivalent ({@code \b \B} outside a class, backreferences, Unicode property
 * escapes, octal escapes) and unknown letter escapes are refused by name
 * ({@link RegexToGbnf.UnsupportedRegexException}); a malformed escape is a
 * {@link RegexToGbnf.InvalidRegexException}.</p>
 */
final class RegexElement {

    /** The decoded code point; -1 for a class shorthand. */
    final int codePoint;
    /** The shorthand letter ({@code d D w W s S}); 0 for a code point. */
    final char shorthand;
    /** How many pattern characters the element spans. */
    final int length;

    private RegexElement(int codePoint, char shorthand, int length) {
        this.codePoint = codePoint;
        this.shorthand = shorthand;
        this.length = length;
    }

    /** Whether the element is a class shorthand rather than one code point. */
    boolean isShorthand() {
        return shorthand != 0;
    }

    /** The element's code points: its shorthand set or the single code point. */
    RegexCharSet toSet() {
        return isShorthand() ? RegexCharSet.shorthand(shorthand) : RegexCharSet.single(codePoint);
    }

    /**
     * Decodes the element starting at {@code pos}.
     *
     * @param pattern the regex text
     * @param pos     index of a plain character or of the backslash of an escape
     * @param inClass whether the element sits inside {@code [ … ]} ({@code \b} is BACKSPACE there)
     * @throws RegexToGbnf.UnsupportedRegexException naming an escape with no grammar equivalent
     * @throws RegexToGbnf.InvalidRegexException     naming a malformed escape
     */
    static RegexElement at(String pattern, int pos, boolean inClass) {
        if (pattern.charAt(pos) != '\\') {
            int codePoint = pattern.codePointAt(pos);
            return new RegexElement(codePoint, (char) 0, Character.charCount(codePoint));
        }
        if (pos + 1 >= pattern.length()) {
            throw new RegexToGbnf.InvalidRegexException("trailing backslash");
        }
        char e = pattern.charAt(pos + 1);
        if (RegexCharSet.isShorthand(e)) {
            return new RegexElement(-1, e, 2);
        }
        switch (e) {
            case 't': return codePoint('\t', 2);
            case 'n': return codePoint('\n', 2);
            case 'r': return codePoint('\r', 2);
            case 'f': return codePoint('\f', 2);
            case 'v': return codePoint(0x0B, 2);
            case '0': return nul(pattern, pos);
            case 'x': return codePoint(hex(pattern, pos + 2, 2, "\\x"), 4);
            case 'u': return unicode(pattern, pos);
            case 'U': return codePoint(hex(pattern, pos + 2, 8, "\\U"), 10);
            case 'c': return control(pattern, pos);
            case 'b':
                if (inClass) {
                    return codePoint('\b', 2);
                }
                throw new RegexToGbnf.UnsupportedRegexException("word boundary assertion: \\b");
            case 'B':
                throw new RegexToGbnf.UnsupportedRegexException("word boundary assertion: \\B");
            case 'k':
                throw new RegexToGbnf.UnsupportedRegexException("backreference: \\k");
            case 'p':
            case 'P':
                throw new RegexToGbnf.UnsupportedRegexException("Unicode property escape: \\" + e);
            default:
                return identity(pattern, pos, e);
        }
    }

    private static RegexElement codePoint(int codePoint, int length) {
        return new RegexElement(codePoint, (char) 0, length);
    }

    /** {@code \0}: NUL, unless a digit follows (an octal escape). */
    private static RegexElement nul(String pattern, int pos) {
        char next = pos + 2 < pattern.length() ? pattern.charAt(pos + 2) : 0;
        if (next >= '0' && next <= '9') {
            throw new RegexToGbnf.UnsupportedRegexException("octal escape: \\0" + next);
        }
        return codePoint(0, 2);
    }

    /** {@code \}{@code uHHHH}, a surrogate pair of two of them, or {@code \}{@code u{H…}}. */
    private static RegexElement unicode(String pattern, int pos) {
        if (pos + 2 < pattern.length() && pattern.charAt(pos + 2) == '{') {
            int close = pattern.indexOf('}', pos + 3);
            int digits = close - (pos + 3);
            if (close < 0 || digits < 1 || digits > 6) {
                throw new RegexToGbnf.InvalidRegexException("invalid \\u{...} escape");
            }
            return codePoint(hex(pattern, pos + 3, digits, "\\u{...}"), close + 1 - pos);
        }
        int unit = hex(pattern, pos + 2, 4, "\\u");
        if (Character.isHighSurrogate((char) unit) && pattern.startsWith("\\u", pos + 6)
                && isHex(pattern, pos + 8, 4)) {
            int low = hex(pattern, pos + 8, 4, "\\u");
            if (Character.isLowSurrogate((char) low)) {
                return codePoint(Character.toCodePoint((char) unit, (char) low), 12);
            }
        }
        return codePoint(unit, 6);
    }

    /** {@code \cX}: the control character of an ASCII letter. */
    private static RegexElement control(String pattern, int pos) {
        char letter = pos + 2 < pattern.length() ? pattern.charAt(pos + 2) : 0;
        if (!((letter >= 'a' && letter <= 'z') || (letter >= 'A' && letter <= 'Z'))) {
            throw new RegexToGbnf.InvalidRegexException("\\c must be followed by an ASCII letter");
        }
        return codePoint(letter % 32, 3);
    }

    /** The identity escape of a character other than an ASCII letter or digit; those refuse. */
    private static RegexElement identity(String pattern, int pos, char e) {
        if (e >= '1' && e <= '9') {
            throw new RegexToGbnf.UnsupportedRegexException("backreference: \\" + e);
        }
        if ((e >= 'a' && e <= 'z') || (e >= 'A' && e <= 'Z')) {
            throw new RegexToGbnf.UnsupportedRegexException("unsupported escape: \\" + e);
        }
        int codePoint = pattern.codePointAt(pos + 1);
        return codePoint(codePoint, 1 + Character.charCount(codePoint));
    }

    /** The {@code digits} hex digits at {@code from} as a code point. */
    private static int hex(String pattern, int from, int digits, String form) {
        if (!isHex(pattern, from, digits)) {
            throw new RegexToGbnf.InvalidRegexException("invalid " + form + " escape");
        }
        long value = Long.parseLong(pattern.substring(from, from + digits), 16);
        if (value > RegexCharSet.MAX_CODE_POINT) {
            throw new RegexToGbnf.InvalidRegexException(form + " escape beyond U+10FFFF");
        }
        return (int) value;
    }

    private static boolean isHex(String pattern, int from, int digits) {
        if (from + digits > pattern.length()) {
            return false;
        }
        for (int i = from; i < from + digits; i++) {
            char h = pattern.charAt(i);
            if (!((h >= '0' && h <= '9') || (h >= 'a' && h <= 'f') || (h >= 'A' && h <= 'F'))) {
                return false;
            }
        }
        return true;
    }
}

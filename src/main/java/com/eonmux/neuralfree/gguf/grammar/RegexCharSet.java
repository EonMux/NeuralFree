package com.eonmux.neuralfree.gguf.grammar;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * An immutable set of Unicode code points kept as sorted, disjoint, non-adjacent inclusive
 * ranges: the character-class model {@link RegexToGbnf} uses for output regexes and JSON-schema
 * patterns. A regex class ({@code [a-z\d]}, {@code [^\s]}) or shorthand ({@code \d \w \s} and
 * the negated {@code \D \W \S}) becomes one set, set union and complement give the class's exact
 * language, and {@link #toGbnf} writes it as one GBNF character set ({@link JsonStringCharacters#set}
 * as JSON-encoded string characters, {@link #toJavaRegex} as a Java class). The shorthand sets are ECMAScript's
 * (ECMA-262 {@code CharacterClassEscape}): {@code \d} is {@code [0-9]}, {@code \w} is
 * {@code [A-Za-z0-9_]} and {@code \s} is the WhiteSpace and LineTerminator code points.
 */
final class RegexCharSet {

    /** The largest Unicode code point. */
    static final int MAX_CODE_POINT = 0x10FFFF;

    /** {@code \d}: the ASCII digits. */
    static final RegexCharSet DIGIT = new RegexCharSet(new int[]{'0', '9'});

    /** {@code \w}: ASCII letters, digits and the underscore. */
    static final RegexCharSet WORD = new RegexCharSet(new int[]{'0', '9', 'A', 'Z', '_', '_', 'a', 'z'});

    /** {@code \s}: ECMAScript WhiteSpace and LineTerminator (TAB, LF, VT, FF, CR, space, NBSP, Zs, LS, PS, BOM). */
    static final RegexCharSet SPACE = new RegexCharSet(new int[]{
        0x09, 0x0D, 0x20, 0x20, 0xA0, 0xA0, 0x1680, 0x1680, 0x2000, 0x200A, 0x2028, 0x2029,
        0x202F, 0x202F, 0x205F, 0x205F, 0x3000, 0x3000, 0xFEFF, 0xFEFF});

    /** The empty set, the start of every union. */
    static final RegexCharSet EMPTY = new RegexCharSet(new int[0]);

    /**
     * ECMAScript's {@code .} without the {@code s} flag: every code point but the
     * LineTerminators LF, CR, LS and PS.
     */
    static final RegexCharSet NON_LINE_TERMINATOR =
            new RegexCharSet(new int[]{0x0A, 0x0A, 0x0D, 0x0D, 0x2028, 0x2029}).complement();

    /** Range bounds, pairs {@code lo, hi}: sorted, disjoint and never adjacent. */
    private final int[] bounds;

    private RegexCharSet(int[] bounds) {
        this.bounds = bounds;
    }

    /**
     * The set of the code points {@code lo..hi}.
     *
     * @throws IllegalArgumentException when the bounds are out of order or outside Unicode
     */
    static RegexCharSet range(int lo, int hi) {
        if (lo < 0 || hi > MAX_CODE_POINT || lo > hi) {
            throw new IllegalArgumentException("invalid code point range " + lo + ".." + hi);
        }
        return new RegexCharSet(new int[]{lo, hi});
    }

    /** The set holding the one code point {@code codePoint}. */
    static RegexCharSet single(int codePoint) {
        return range(codePoint, codePoint);
    }

    /**
     * The set of a class shorthand letter, the negated forms ({@code D W S}) already
     * complemented.
     *
     * @throws IllegalArgumentException when {@code letter} is not one of {@code d D w W s S}
     */
    static RegexCharSet shorthand(char letter) {
        switch (letter) {
            case 'd': return DIGIT;
            case 'D': return DIGIT.complement();
            case 'w': return WORD;
            case 'W': return WORD.complement();
            case 's': return SPACE;
            case 'S': return SPACE.complement();
            default:
                throw new IllegalArgumentException("not a class shorthand: \\" + letter);
        }
    }

    /** Whether {@code letter} names a class shorthand ({@code d D w W s S}). */
    static boolean isShorthand(char letter) {
        return "dDwWsS".indexOf(letter) >= 0;
    }

    /** Whether the set holds no code point. */
    boolean isEmpty() {
        return bounds.length == 0;
    }

    /** The union of this set and {@code other}, a new set. */
    RegexCharSet union(RegexCharSet other) {
        int[] all = Arrays.copyOf(bounds, bounds.length + other.bounds.length);
        System.arraycopy(other.bounds, 0, all, bounds.length, other.bounds.length);
        return new RegexCharSet(normalize(all));
    }

    /** Whether the set holds {@code codePoint}. */
    boolean contains(int codePoint) {
        for (int i = 0; i < bounds.length; i += 2) {
            if (codePoint <= bounds[i + 1]) {
                return codePoint >= bounds[i];
            }
        }
        return false;
    }

    /**
     * This set with the other case of every ASCII letter it holds added, a new set: the code points an
     * ASCII-case-insensitive match of any member accepts, as xgrammar's {@code i} flag folds ({@code k} and
     * {@code K}; no other code point has a variant).
     */
    RegexCharSet asciiCaseClosure() {
        int[] added = new int[4 * bounds.length];
        int count = 0;
        for (int i = 0; i < bounds.length; i += 2) {
            for (int[] letters : new int[][]{{'A', 'Z'}, {'a', 'z'}}) {
                int lo = Math.max(bounds[i], letters[0]);
                int hi = Math.min(bounds[i + 1], letters[1]);
                if (lo <= hi) {
                    added[count++] = lo ^ 0x20;
                    added[count++] = hi ^ 0x20;
                }
            }
        }
        return count == 0 ? this : union(new RegexCharSet(normalize(Arrays.copyOf(added, count))));
    }

    /** The range bounds as pairs {@code lo, hi}, sorted, disjoint and never adjacent; a copy. */
    int[] bounds() {
        return bounds.clone();
    }

    /** The set of the ranges {@code pairs} ({@code lo, hi, ...}, in any order, overlaps allowed). */
    static RegexCharSet ofRanges(int[] pairs) {
        if (pairs.length % 2 != 0) {
            throw new IllegalArgumentException("ranges come in lo, hi pairs");
        }
        for (int i = 0; i < pairs.length; i += 2) {
            if (pairs[i] < 0 || pairs[i + 1] > MAX_CODE_POINT || pairs[i] > pairs[i + 1]) {
                throw new IllegalArgumentException("invalid code point range " + pairs[i] + ".." + pairs[i + 1]);
            }
        }
        return new RegexCharSet(normalize(pairs.clone()));
    }

    /** The code points of this set outside {@code other}, a new set. */
    RegexCharSet minus(RegexCharSet other) {
        return complement().union(other).complement();
    }

    /** The code points of {@code 0..MAX_CODE_POINT} outside this set, a new set. */
    RegexCharSet complement() {
        List<Integer> out = new ArrayList<>();
        int next = 0;
        for (int i = 0; i < bounds.length; i += 2) {
            if (bounds[i] > next) {
                out.add(next);
                out.add(bounds[i] - 1);
            }
            next = bounds[i + 1] + 1;
        }
        if (next <= MAX_CODE_POINT) {
            out.add(next);
            out.add(MAX_CODE_POINT);
        }
        int[] result = new int[out.size()];
        for (int i = 0; i < result.length; i++) {
            result[i] = out.get(i);
        }
        return new RegexCharSet(result);
    }

    /**
     * The set as one GBNF character set: {@code [ranges]}, or {@code [^ranges]} when
     * {@code negated}, each bound written by {@link #classChar}.
     *
     * @throws IllegalStateException when the set is empty (GBNF has no empty character set)
     */
    String toGbnf(boolean negated) {
        if (isEmpty()) {
            throw new IllegalStateException("an empty character set has no GBNF form");
        }
        StringBuilder out = new StringBuilder(negated ? "[^" : "[");
        for (int i = 0; i < bounds.length; i += 2) {
            out.append(classChar(bounds[i]));
            if (bounds[i + 1] != bounds[i]) {
                out.append('-').append(classChar(bounds[i + 1]));
            }
        }
        return out.append(']').toString();
    }

    /**
     * The set as one {@code java.util.regex} character class, each bound written
     * {@code \x{H}}; the empty set is a class no code point matches.
     */
    String toJavaRegex() {
        if (isEmpty()) {
            return "[^\\x{0}-\\x{10FFFF}]";
        }
        StringBuilder out = new StringBuilder("[");
        for (int i = 0; i < bounds.length; i += 2) {
            out.append(javaChar(bounds[i]));
            if (bounds[i + 1] != bounds[i]) {
                out.append('-').append(javaChar(bounds[i + 1]));
            }
        }
        return out.append(']').toString();
    }

    /**
     * One code point in {@code java.util.regex} syntax, inside or outside a class: an ASCII
     * letter or digit raw, any other code point as {@code \x{H}}.
     */
    static String javaChar(int codePoint) {
        boolean plain = (codePoint >= '0' && codePoint <= '9') || (codePoint >= 'a' && codePoint <= 'z')
                || (codePoint >= 'A' && codePoint <= 'Z');
        return plain ? String.valueOf((char) codePoint) : "\\x{" + Integer.toHexString(codePoint) + "}";
    }

    /**
     * One code point inside a GBNF character set: the set's metacharacters ({@code \ ] - [ ^})
     * escaped, the rest as {@link #literalChar} writes it.
     */
    static String classChar(int codePoint) {
        switch (codePoint) {
            case '\\': return "\\\\";
            case ']': return "\\]";
            case '[': return "\\[";
            case '-': return "\\-";
            case '^': return "\\x5E";
            default:
                return literalChar(codePoint);
        }
    }

    /**
     * One code point inside a GBNF string literal: {@code "} and {@code \} escaped, TAB, LF and
     * CR as {@code \t \n \r}, other C0/C1 controls and DEL as {@code \xNN}, lone surrogates as
     * {@code \}{@code uNNNN}, everything else raw.
     */
    static String literalChar(int codePoint) {
        switch (codePoint) {
            case '"': return "\\\"";
            case '\\': return "\\\\";
            case '\t': return "\\t";
            case '\n': return "\\n";
            case '\r': return "\\r";
            default:
                break;
        }
        if (codePoint < 0x20 || (codePoint >= 0x7F && codePoint <= 0x9F)) {
            return String.format("\\x%02X", codePoint);
        }
        if (codePoint >= Character.MIN_SURROGATE && codePoint <= Character.MAX_SURROGATE) {
            return String.format("\\u%04X", codePoint);
        }
        return new String(Character.toChars(codePoint));
    }

    /** Sorts {@code pairs} by lower bound and merges overlapping or adjacent ranges. */
    private static int[] normalize(int[] pairs) {
        int n = pairs.length / 2;
        long[] packed = new long[n];
        for (int i = 0; i < n; i++) {
            packed[i] = ((long) pairs[2 * i] << 32) | (pairs[2 * i + 1] & 0xFFFFFFFFL);
        }
        Arrays.sort(packed);
        List<Integer> out = new ArrayList<>();
        for (long p : packed) {
            int lo = (int) (p >>> 32);
            int hi = (int) p;
            int last = out.size() - 1;
            if (last >= 0 && lo <= out.get(last) + 1) {
                out.set(last, Math.max(out.get(last), hi));
            } else {
                out.add(lo);
                out.add(hi);
            }
        }
        int[] result = new int[out.size()];
        for (int i = 0; i < result.length; i++) {
            result[i] = out.get(i);
        }
        return result;
    }
}

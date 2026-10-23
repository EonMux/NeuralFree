package com.eonmux.neuralfree.gguf.tokenizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;

/**
 * Java port of llama.cpp {@code unicode_regex_split} (src/unicode.cpp:1484-1680) without the
 * byte encoding: an ordered list of regex steps, each splitting every current word into its
 * matches and the unmatched text between them.
 *
 * <p>A step whose regex text equals one of the expressions llama.cpp implements by hand
 * ({@code unicode_regex_split_custom}, src/unicode.cpp:1219-1259) runs the ported splitter in
 * {@link UnicodeCustomSplits}. Every other step runs as a Java regex with llama.cpp's
 * semantics:</p>
 * <ul>
 *   <li>A regex that names a Unicode category ({@code \p{N}}, {@code \p{L}}, {@code \p{P}},
 *       {@code \p{M}}, {@code \p{S}}, {@code \p{Lu}}, {@code \p{Ll}}, {@code \p{Lt}},
 *       {@code \p{Lm}}, {@code \p{Lo}}) runs on the "collapsed" text, where each non-ASCII code
 *       point becomes one placeholder per category ({@code 0xD0}-{@code 0xD5}) and non-ASCII
 *       whitespace becomes U+000B; each category in the regex becomes its placeholder plus its
 *       ASCII members. All letter sub-categories therefore mean {@code \p{L}}, as in llama.cpp.</li>
 *   <li>Any other regex runs on the code points, with non-ASCII whitespace replaced by U+000B.</li>
 *   <li>Classes {@code \s}, {@code \S} and {@code \d} are ASCII-only (no
 *       {@link Pattern#UNICODE_CHARACTER_CLASS}), matching {@code std::regex} in the C locale.
 *       A top-level {@code $} means end of word ({@code \z}).</li>
 *   <li>Each word is matched on its own: lookaheads and anchors stop at the word boundary
 *       ({@code std::regex_iterator} over the word's range).</li>
 * </ul>
 */
final class UnicodeRegexSplit {

    /** llama.cpp {@code k_ucat_enum}: the category escapes that trigger the collapsed text. */
    private static final Map<String, Integer> CATEGORY_ESCAPES = categoryEscapes();

    /*
     * java.util.regex recurses once per iteration of a repeated group, so two group forms of the
     * llama.cpp table are rewritten to equivalent forms whose long runs repeat a single class.
     * Both rewrites accept the same language with the same greedy preference.
     */

    /** EXAONE_MOE letter words joined by single spaces: {@code (U( U)*)+} with U = {@code \p{L}\p{M}*}. */
    private static final String LETTER_WORDS_GROUPED = "(?:\\p{L}\\p{M}*(?: \\p{L}\\p{M}*)*)+";
    /** The same language: a letter, then letters or marks, with single spaces before letters. */
    private static final String LETTER_WORDS_FLAT = "\\p{L}[\\p{L}\\p{M}]*(?: \\p{L}[\\p{L}\\p{M}]*)*";
    /** TEKKEN/GPT4O one-character units {@code ((?=[A])([^B]))}, rewritten as {@code [[A]&&[^B]]}. */
    private static final Pattern LOOKAHEAD_CLASS =
            Pattern.compile("\\(\\(\\?=\\[([^\\]\\[]*)\\]\\)\\(\\[\\^([^\\]\\[]*)\\]\\)\\)");

    private UnicodeRegexSplit() {}

    /** One compiled split step. Implementations are immutable. */
    interface Step {
        /** Splits every word of {@code offsets} (lengths in code points) into finer words. */
        Offsets apply(Text text, Offsets offsets);
    }

    /**
     * Compiles a llama.cpp regex sequence.
     *
     * @throws IllegalArgumentException when a regex mixes categories with non-ASCII characters
     *         or is not valid regex syntax
     */
    static List<Step> compile(List<String> regexes) {
        List<Step> steps = new ArrayList<>(regexes.size());
        for (String r : regexes) {
            Step custom = UnicodeCustomSplits.forRegex(r);
            steps.add(custom != null ? custom : RegexStep.of(r));
        }
        return Collections.unmodifiableList(steps);
    }

    /** Splits text into words; an empty text yields no words, and empty matches yield no word. */
    static List<String> split(String text, List<Step> steps) {
        List<String> words = new ArrayList<>();
        if (text.isEmpty()) return words;
        Text t = new Text(text);
        Offsets offsets = new Offsets(1);
        offsets.add(t.cps.length);
        for (Step s : steps) offsets = s.apply(t, offsets);
        int start = 0;
        for (int i = 0; i < offsets.size(); i++) {
            int len = offsets.get(i);
            if (len > 0) words.add(text.substring(t.cpToChar[start], t.cpToChar[start + len]));
            start += len;
        }
        return words;
    }

    // ---------------------------------------------------------------------------------------------
    // Text and offsets
    // ---------------------------------------------------------------------------------------------

    /** The code points of the text being split and its lazily built regex views. */
    static final class Text {
        final String text;
        final int[] cps;
        final int[] flags;
        /** Char index of each code point, plus the text length at index {@code cps.length}. */
        final int[] cpToChar;
        private String collapsed;
        private String wide;

        Text(String text) {
            this.text = text;
            this.cps = text.codePoints().toArray();
            this.flags = UnicodeFlags.of(cps);
            this.cpToChar = new int[cps.length + 1];
            int c = 0;
            for (int i = 0; i < cps.length; i++) {
                cpToChar[i] = c;
                c += Character.charCount(cps[i]);
            }
            cpToChar[cps.length] = c;
        }

        /** One char per code point: ASCII as is, non-ASCII whitespace U+000B, else the category placeholder. */
        String collapsed() {
            if (collapsed == null) {
                char[] out = new char[cps.length];
                for (int i = 0; i < cps.length; i++) {
                    if (cps[i] < 128) out[i] = (char) cps[i];
                    else if ((flags[i] & UnicodeFlags.WHITESPACE) != 0) out[i] = 0x0B;
                    else out[i] = placeholder(flags[i] & UnicodeFlags.MASK_CATEGORIES);
                }
                collapsed = new String(out);
            }
            return collapsed;
        }

        /** The text with non-ASCII whitespace replaced by U+000B (all such code points are BMP). */
        String wide() {
            if (wide == null) {
                char[] out = text.toCharArray();
                for (int i = 0; i < cps.length; i++) {
                    if (cps[i] > 0x7F && (flags[i] & UnicodeFlags.WHITESPACE) != 0) out[cpToChar[i]] = 0x0B;
                }
                wide = new String(out);
            }
            return wide;
        }
    }

    /** Growable list of word lengths in code points. */
    static final class Offsets {
        private int[] data;
        private int size;

        Offsets(int capacity) {
            this.data = new int[Math.max(capacity, 4)];
        }

        void add(int length) {
            if (size == data.length) data = Arrays.copyOf(data, data.length * 2);
            data[size++] = length;
        }

        int get(int i) { return data[i]; }
        int size() { return size; }
    }

    // ---------------------------------------------------------------------------------------------
    // Generic regex steps
    // ---------------------------------------------------------------------------------------------

    /** {@code unicode_regex_split_stl} over the collapsed or the wide text. */
    private static final class RegexStep implements Step {
        private final Pattern pattern;
        private final boolean collapsed;

        private RegexStep(Pattern pattern, boolean collapsed) {
            this.pattern = pattern;
            this.collapsed = collapsed;
        }

        static RegexStep of(String regex) {
            boolean collapsed = usesCategories(regex);
            String ecma = regex.replace(LETTER_WORDS_GROUPED, LETTER_WORDS_FLAT);
            if (collapsed) {
                for (int i = 0; i < regex.length(); i++) {
                    if (regex.charAt(i) >= 128) {
                        throw new IllegalArgumentException("Regex includes both unicode categories and non-ASCII"
                                + " characters - not supported: '" + regex + "'");
                    }
                }
                ecma = collapseCategories(ecma);
            }
            try {
                return new RegexStep(Pattern.compile(flattenLookaheadClasses(toJavaSyntax(ecma))), collapsed);
            } catch (PatternSyntaxException e) {
                throw new IllegalArgumentException("Failed to process pre-tokenizer regex '" + regex + "': "
                        + e.getDescription(), e);
            }
        }

        @Override
        public Offsets apply(Text t, Offsets offsets) {
            String s = collapsed ? t.collapsed() : t.wide();
            Matcher m = pattern.matcher(s);
            Offsets out = new Offsets(offsets.size() * 2);
            int start = 0;
            for (int k = 0; k < offsets.size(); k++) {
                int len = offsets.get(k);
                int from = charIndex(t, start);
                int to = charIndex(t, start + len);
                m.region(from, to);
                int prev = from;
                while (m.find()) {
                    if (m.start() > prev) out.add(cpCount(t, s, prev, m.start()));
                    out.add(cpCount(t, s, m.start(), m.end()));
                    prev = m.end();
                }
                if (prev < to) out.add(cpCount(t, s, prev, to));
                start += len;
            }
            return out;
        }

        private int charIndex(Text t, int cp) {
            return collapsed ? cp : t.cpToChar[cp];
        }

        private int cpCount(Text t, String s, int fromChar, int toChar) {
            return collapsed ? toChar - fromChar : Character.codePointCount(s, fromChar, toChar);
        }
    }

    private static boolean usesCategories(String regex) {
        for (String esc : CATEGORY_ESCAPES.keySet()) {
            if (regex.contains(esc)) return true;
        }
        return false;
    }

    /**
     * The llama.cpp collapsed-regex rewrite: each category escape becomes its placeholder plus its
     * ASCII members, wrapped in a class when it stands outside one.
     */
    static String collapseCategories(String re) {
        StringBuilder out = new StringBuilder(re.length() * 2);
        boolean inside = false;
        for (int i = 0; i < re.length(); i++) {
            char c = re.charAt(i);
            if (c == '[' && (i == 0 || re.charAt(i - 1) != '\\')) {
                out.append('[');
                inside = true;
                continue;
            }
            if (inside && c == ']' && re.charAt(i - 1) != '\\') {
                out.append(']');
                inside = false;
                continue;
            }
            if (c == '\\' && i + 3 < re.length() && re.charAt(i + 1) == 'p' && re.charAt(i + 2) == '{') {
                int close = re.indexOf('}', i + 3);
                if (close >= 0 && close <= i + 10) {
                    Integer flag = CATEGORY_ESCAPES.get(re.substring(i, close + 1));
                    if (flag != null) {
                        if (!inside) out.append('[');
                        out.append(String.format("\\x{%02X}", (int) placeholder(flag))).append(asciiMembers(flag));
                        if (!inside) out.append(']');
                        i = close;
                        continue;
                    }
                }
            }
            out.append(c);
        }
        return out.toString();
    }

    /**
     * Rewrites the ECMAScript constructs whose Java meaning differs: inside a class {@code [} and
     * {@code &} are literals; outside one, {@code $} is the end of the input.
     */
    static String toJavaSyntax(String re) {
        StringBuilder out = new StringBuilder(re.length() + 8);
        boolean inClass = false;
        for (int i = 0; i < re.length(); i++) {
            char c = re.charAt(i);
            if (c == '\\') {
                out.append(c);
                if (i + 1 < re.length()) out.append(re.charAt(++i));
                continue;
            }
            if (inClass) {
                if (c == ']') {
                    inClass = false;
                    out.append(c);
                } else if (c == '[' || c == '&') {
                    out.append('\\').append(c);
                } else {
                    out.append(c);
                }
                continue;
            }
            if (c == '[') {
                inClass = true;
                out.append(c);
                if (i + 1 < re.length() && re.charAt(i + 1) == '^') out.append(re.charAt(++i));
                if (i + 1 < re.length() && re.charAt(i + 1) == ']') {
                    throw new IllegalArgumentException("Empty character class is not supported in '" + re + "'");
                }
            } else if (c == '$') {
                out.append("\\z");
            } else {
                out.append(c);
            }
        }
        return out.toString();
    }

    /**
     * Replaces each one-character unit {@code ((?=[A])([^B]))} of a Java-syntax regex with the
     * class intersection {@code [[A]&&[^B]]}: both match exactly one character that is in A and
     * not in B, and the regexes using them never refer to the groups.
     */
    static String flattenLookaheadClasses(String re) {
        Matcher m = LOOKAHEAD_CLASS.matcher(re);
        StringBuffer out = new StringBuffer(re.length());
        while (m.find()) {
            m.appendReplacement(out, Matcher.quoteReplacement("[[" + m.group(1) + "]&&[^" + m.group(2) + "]]"));
        }
        m.appendTail(out);
        return out.toString();
    }

    /** llama.cpp {@code k_ucat_cpt}: the collapsed placeholder of a category. */
    private static char placeholder(int category) {
        switch (category) {
            case UnicodeFlags.NUMBER: return (char) 0xD1;
            case UnicodeFlags.LETTER: return (char) 0xD2;
            case UnicodeFlags.PUNCTUATION: return (char) 0xD3;
            case UnicodeFlags.ACCENT_MARK: return (char) 0xD4;
            case UnicodeFlags.SYMBOL: return (char) 0xD5;
            default: return (char) 0xD0;
        }
    }

    /** llama.cpp {@code k_ucat_map}: the ASCII members of a category, as class content. */
    private static String asciiMembers(int category) {
        switch (category) {
            case UnicodeFlags.NUMBER: return "0-9";
            case UnicodeFlags.LETTER: return "A-Za-z";
            case UnicodeFlags.PUNCTUATION: return "!-#%-*,-/:-;?-@\\[-\\]_\\{\\}";
            case UnicodeFlags.SYMBOL: return "\\$\\+<->\\^`\\|\\~";
            default: return "";
        }
    }

    private static Map<String, Integer> categoryEscapes() {
        Map<String, Integer> m = new LinkedHashMap<>();
        m.put("\\p{N}", UnicodeFlags.NUMBER);
        m.put("\\p{L}", UnicodeFlags.LETTER);
        m.put("\\p{P}", UnicodeFlags.PUNCTUATION);
        m.put("\\p{M}", UnicodeFlags.ACCENT_MARK);
        m.put("\\p{S}", UnicodeFlags.SYMBOL);
        m.put("\\p{Lu}", UnicodeFlags.LETTER);
        m.put("\\p{Ll}", UnicodeFlags.LETTER);
        m.put("\\p{Lt}", UnicodeFlags.LETTER);
        m.put("\\p{Lm}", UnicodeFlags.LETTER);
        m.put("\\p{Lo}", UnicodeFlags.LETTER);
        return Collections.unmodifiableMap(m);
    }
}

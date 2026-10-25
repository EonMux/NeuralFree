package com.eonmux.neuralfree.gguf.tokenizer;

import com.eonmux.neuralfree.gguf.tokenizer.PreTokenizerSpec.Behaviour;
import com.eonmux.neuralfree.gguf.tokenizer.PreTokenizerSpec.Dialect;
import com.eonmux.neuralfree.gguf.tokenizer.PreTokenizerSpec.PrependScheme;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.function.IntPredicate;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;

/**
 * Java port of llama.cpp {@code unicode_regex_split} (src/unicode.cpp:1484-1680) without the
 * byte encoding, extended with the Hugging Face pre-tokenizer steps of {@link PreTokenizerSpec}:
 * an ordered list of steps, each splitting every current word into its matches and the unmatched
 * text between them.
 *
 * <p>A step whose regex text equals one of the expressions llama.cpp implements by hand
 * ({@code unicode_regex_split_custom}, src/unicode.cpp:1219-1259) runs the ported splitter in
 * {@link UnicodeCustomSplits}. Every other {@link Dialect#LLAMA_CPP} step runs as a Java regex
 * with llama.cpp's semantics:</p>
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
 *
 * <p>{@link Dialect#ONIGURUMA} steps run on the code points with Unicode classes and Oniguruma's
 * syntax translated to Java ({@link #onigurumaToJava}). Split behaviours, inversion and the
 * empty-match rule follow Hugging Face {@code NormalizedString::split}.</p>
 *
 * <p>Every regex match call on a pattern taken from a model file reads its text through a
 * {@link BudgetedCharSequence}. The patterns NeuralFree itself ships, the llama.cpp and Hugging
 * Face forms of every {@link PreTokenizerType} and the fixed {@code Whitespace} pattern, are
 * matched directly, as llama.cpp and Hugging Face match them: they are reviewed constants, and
 * a legitimately super-linear one (such as {@code \s+$} over a long run of spaces) must not
 * reject input that llama.cpp tokenizes.</p>
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

    /**
     * Java flags of the Oniguruma dialect: Unicode {@code \s\w\d\b}, and {@code ^}/{@code $} as
     * line anchors with {@code \n} as the only line terminator (Oniguruma's default options).
     */
    static final int ONIGURUMA_FLAGS = Pattern.UNICODE_CHARACTER_CLASS | Pattern.MULTILINE | Pattern.UNIX_LINES;

    /** HF {@code Whitespace}: the runs it keeps. */
    private static final String WHITESPACE_WORDS = "\\w+|[^\\w\\s]+";
    private static final Pattern WHITESPACE_WORDS_PATTERN =
            Pattern.compile(WHITESPACE_WORDS, Pattern.UNICODE_CHARACTER_CLASS);

    /** The {@link Dialect#LLAMA_CPP} pattern texts NeuralFree ships: every {@link PreTokenizerType#regexes()}. */
    private static final Set<String> BUILT_IN_LLAMA_CPP = builtInRegexes(false);
    /** The {@link Dialect#ONIGURUMA} pattern texts NeuralFree ships: every {@link PreTokenizerType#hfRegexes()}. */
    private static final Set<String> BUILT_IN_ONIGURUMA = builtInRegexes(true);

    /** Oniguruma general-category names (short and long, normalized) to Java's short names. */
    private static final Map<String, String> GENERAL_CATEGORIES = generalCategories();
    /** Oniguruma POSIX property names (normalized) to Java's POSIX names. */
    private static final Map<String, String> POSIX_PROPERTIES = posixProperties();
    private static final String ANY_CLASS = "[\\x{0}-\\x{10FFFF}]";
    private static final String NO_CLASS = "[^\\x{0}-\\x{10FFFF}]";
    /** {@link #fixedScript} of a space, which continues any script. */
    private static final int SCRIPT_ANY = -2;

    private UnicodeRegexSplit() {}

    /** One compiled split step. Implementations are immutable. */
    interface Step {
        /**
         * Splits every word of {@code offsets} (lengths in code points) into finer words. A step
         * may mark words removed, and a step that rewrites the words returns offsets over its
         * new text ({@link Offsets#over}).
         */
        Offsets apply(Text text, Offsets offsets);
    }

    /**
     * Compiles a pre-tokenizer spec. When {@link PreTokenizerSpec#encodesBeforeLastStep()} is
     * set, the words are byte-encoded where the BYTE_LEVEL step stands; otherwise byte encoding
     * is left to the caller.
     *
     * @throws IllegalArgumentException when a regex cannot be compiled in its dialect
     */
    static List<Step> compile(PreTokenizerSpec spec) {
        List<Step> steps = new ArrayList<>(spec.steps().size() + 2);
        List<PreTokenizerSpec.Step> specSteps = spec.steps();
        for (int i = 0; i < specSteps.size(); i++) {
            PreTokenizerSpec.Step s = specSteps.get(i);
            switch (s.kind()) {
                case REGEX_SPLIT:
                    steps.add(regexStep(s.pattern(), s.dialect(), s.behaviour(), s.invert()));
                    break;
                case WHITESPACE:
                    steps.add(new RegexStep(WHITESPACE_WORDS_PATTERN, WHITESPACE_WORDS, false, View.RAW,
                            Behaviour.REMOVED, true));
                    break;
                case WHITESPACE_SPLIT:
                    steps.add(new PredicateStep(UnicodeRegexSplit::isWhiteSpace, Behaviour.REMOVED));
                    break;
                case PUNCTUATION:
                    steps.add(new PredicateStep(UnicodeRegexSplit::isPunctuation, s.behaviour()));
                    break;
                case CHAR_DELIMITER:
                    int delimiter = s.delimiter();
                    steps.add(new PredicateStep(cp -> cp == delimiter, Behaviour.REMOVED));
                    break;
                case DIGITS:
                    steps.add(new PredicateStep(UnicodeRegexSplit::isNumeric,
                            s.individualDigits() ? Behaviour.ISOLATED : Behaviour.CONTIGUOUS));
                    break;
                case BYTE_LEVEL:
                    if (spec.addPrefixSpace()) steps.add(UnicodeRegexSplit::prefixSpace);
                    if (spec.useRegex()) steps.add(llamaCppIsolated(PreTokenizerPatterns.GPT2.get(0)));
                    if (i < specSteps.size() - 1) steps.add(UnicodeRegexSplit::byteLevelEncode);
                    break;
                case METASPACE:
                    steps.add(new MetaspaceStep(s.replacement(), s.prependScheme(), s.split()));
                    break;
                case UNICODE_SCRIPTS:
                    steps.add(UnicodeRegexSplit::unicodeScripts);
                    break;
                case BERT:
                    steps.add(new PredicateStep(UnicodeRegexSplit::isWhiteSpace, Behaviour.REMOVED));
                    steps.add(new PredicateStep(UnicodeRegexSplit::isPunctuation, Behaviour.ISOLATED));
                    break;
                default:
                    throw new IllegalStateException("unhandled pre-tokenizer step kind " + s.kind());
            }
        }
        return Collections.unmodifiableList(steps);
    }

    /**
     * Splits text into words; an empty text yields no words, and empty matches yield no word.
     * {@code inputStart} tells whether the text begins the whole input, which only the
     * {@link PrependScheme#FIRST} Metaspace scheme reads.
     */
    static List<String> split(String text, List<Step> steps, boolean inputStart) {
        List<String> words = new ArrayList<>();
        if (text.isEmpty()) return words;
        Text t = new Text(text, inputStart);
        Offsets offsets = new Offsets(1);
        offsets.add(t.cps.length);
        for (Step s : steps) {
            offsets = s.apply(t, offsets);
            if (offsets.text != null) t = offsets.text;
            if (offsets.removed > 0) {
                offsets = compact(t, offsets);
                t = offsets.text;
            }
        }
        int start = 0;
        for (int i = 0; i < offsets.size(); i++) {
            int len = offsets.get(i);
            if (len > 0) words.add(t.text.substring(t.cpToChar[start], t.cpToChar[start + len]));
            start += len;
        }
        return words;
    }

    /** The kept non-empty words as a new text. */
    private static Offsets compact(Text t, Offsets offsets) {
        StringBuilder sb = new StringBuilder(t.text.length());
        Offsets out = new Offsets(offsets.size());
        boolean leading = t.atInputStart;
        boolean seenWord = false;
        int start = 0;
        for (int i = 0; i < offsets.size(); i++) {
            int len = offsets.get(i);
            if (len > 0) {
                if (offsets.isRemoved(i)) {
                    if (!seenWord) leading = false;
                } else {
                    sb.append(t.text, t.cpToChar[start], t.cpToChar[start + len]);
                    out.add(len);
                }
                seenWord = true;
            }
            start += len;
        }
        return out.over(new Text(sb.toString(), leading));
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
        /** The first word of this text starts the whole input. */
        final boolean atInputStart;
        private String collapsed;
        private String wide;

        Text(String text, boolean atInputStart) {
            this.text = text;
            this.atInputStart = atInputStart;
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

    /**
     * Growable list of word lengths in code points. A word may be marked removed; offsets
     * returned by a rewriting step carry the step's new text.
     */
    static final class Offsets {
        private int[] data;
        private int size;
        private int removed;
        private Text text;

        Offsets(int capacity) {
            this.data = new int[Math.max(capacity, 4)];
        }

        void add(int length) {
            if (size == data.length) data = Arrays.copyOf(data, data.length * 2);
            data[size++] = length;
        }

        /** Adds a word that is dropped from the result. */
        void addRemoved(int length) {
            add(~length);
            removed++;
        }

        /** These offsets describe {@code newText} instead of the text the step received. */
        Offsets over(Text newText) {
            this.text = newText;
            return this;
        }

        int get(int i) {
            int v = data[i];
            return v < 0 ? ~v : v;
        }

        boolean isRemoved(int i) { return data[i] < 0; }
        int size() { return size; }
    }

    // ---------------------------------------------------------------------------------------------
    // Regex steps
    // ---------------------------------------------------------------------------------------------

    /** The string a regex step matches against. */
    private enum View { COLLAPSED, WIDE, RAW }

    private static Step regexStep(String regex, Dialect dialect, Behaviour behaviour, boolean invert) {
        if (dialect == Dialect.ONIGURUMA) {
            return new RegexStep(compileOniguruma(regex), regex, !BUILT_IN_ONIGURUMA.contains(regex), View.RAW,
                    behaviour, invert);
        }
        if (behaviour == Behaviour.ISOLATED) return llamaCppIsolated(regex);
        try {
            return RegexStep.llamaCpp(regex, behaviour, invert);
        } catch (IllegalArgumentException e) {
            if (UnicodeCustomSplits.forRegex(regex) == null) throw e;
            throw new IllegalArgumentException("llama.cpp regex '" + regex + "' is implemented by a hand-written"
                    + " splitter, which supports only the ISOLATED behaviour, and is not valid as a generic regex: "
                    + e.getMessage(), e);
        }
    }

    /** {@code unicode_regex_split}: the custom splitter for its exact text, else the generic regex. */
    private static Step llamaCppIsolated(String regex) {
        Step custom = UnicodeCustomSplits.forRegex(regex);
        return custom != null ? custom : RegexStep.llamaCpp(regex, Behaviour.ISOLATED, false);
    }

    /**
     * A Java regex over one view of the text. ISOLATED llama.cpp steps run
     * {@code unicode_regex_split_stl}; every other step builds the Hugging Face match list and
     * applies its behaviour. A budgeted step matches through a {@link BudgetedCharSequence}.
     */
    private static final class RegexStep implements Step {
        private final Pattern pattern;
        private final String source;
        private final boolean budgeted;
        private final View view;
        private final Behaviour behaviour;
        private final boolean invert;

        RegexStep(Pattern pattern, String source, boolean budgeted, View view, Behaviour behaviour, boolean invert) {
            this.pattern = pattern;
            this.source = source;
            this.budgeted = budgeted;
            this.view = view;
            this.behaviour = behaviour;
            this.invert = invert;
        }

        static RegexStep llamaCpp(String regex, Behaviour behaviour, boolean invert) {
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
                return new RegexStep(Pattern.compile(flattenLookaheadClasses(toJavaSyntax(ecma))), regex,
                        !BUILT_IN_LLAMA_CPP.contains(regex), collapsed ? View.COLLAPSED : View.WIDE, behaviour, invert);
            } catch (PatternSyntaxException e) {
                throw new IllegalArgumentException("Failed to process pre-tokenizer regex '" + regex + "': "
                        + e.getDescription(), e);
            }
        }

        @Override
        public Offsets apply(Text t, Offsets offsets) {
            String s = view == View.COLLAPSED ? t.collapsed() : view == View.WIDE ? t.wide() : t.text;
            BudgetedCharSequence seq = budgeted ? new BudgetedCharSequence(s, source) : null;
            Matcher m = pattern.matcher(seq != null ? seq : s);
            Offsets out = new Offsets(offsets.size() * 2);
            Segments segments = view == View.RAW || behaviour != Behaviour.ISOLATED ? new Segments() : null;
            int start = 0;
            for (int k = 0; k < offsets.size(); k++) {
                int len = offsets.get(k);
                int from = charIndex(t, start);
                int to = charIndex(t, start + len);
                m.region(from, to);
                if (segments == null) {
                    isolated(t, s, m, seq, from, to, out);
                } else {
                    segments.clear();
                    int last = from;
                    int lastMatchEnd = -1;
                    while (find(m, seq, source, to - from)) {
                        int ms = m.start();
                        int me = m.end();
                        if (ms == me && ms == lastMatchEnd) continue;
                        if (view != View.COLLAPSED && splitsSurrogatePair(s, ms)) continue;
                        if (ms > last) segments.add(cpCount(t, s, last, ms), false);
                        segments.add(cpCount(t, s, ms, me), true);
                        last = me;
                        lastMatchEnd = me;
                    }
                    if (last < to) segments.add(cpCount(t, s, last, to), false);
                    segments.emit(behaviour, invert, out);
                }
                start += len;
            }
            return out;
        }

        /** {@code unicode_regex_split_stl}: every match and every gap is a word. */
        private void isolated(Text t, String s, Matcher m, BudgetedCharSequence seq, int from, int to, Offsets out) {
            int prev = from;
            while (find(m, seq, source, to - from)) {
                if (m.start() > prev) out.add(cpCount(t, s, prev, m.start()));
                out.add(cpCount(t, s, m.start(), m.end()));
                prev = m.end();
            }
            if (prev < to) out.add(cpCount(t, s, prev, to));
        }

        private int charIndex(Text t, int cp) {
            return view == View.COLLAPSED ? cp : t.cpToChar[cp];
        }

        private int cpCount(Text t, String s, int fromChar, int toChar) {
            return view == View.COLLAPSED ? toChar - fromChar : Character.codePointCount(s, fromChar, toChar);
        }
    }

    /**
     * One {@link Matcher#find()} call of {@code pattern} over an input of {@code inputLength}
     * chars, with a fresh read budget when the matcher reads {@code budget} (null for an
     * unbudgeted built-in pattern).
     *
     * @throws IllegalStateException when the match exceeds its read budget or the stack
     */
    static boolean find(Matcher m, BudgetedCharSequence budget, String pattern, int inputLength) {
        if (budget != null) budget.reset(inputLength);
        try {
            return m.find();
        } catch (StackOverflowError e) {
            throw new IllegalStateException("tokenizer regex '" + pattern + "' exhausted the stack on a "
                    + inputLength + "-char input", e);
        }
    }

    /**
     * The Hugging Face match list of {@code pattern} over {@code text}: {@code [start, end)} char
     * pairs in order, with an empty match directly after the previous match skipped (the
     * {@code onig} crate's {@code find_iter}). An empty text has no matches, as in Hugging Face
     * {@code find_matches}.
     */
    static int[] findMatches(Pattern pattern, String source, String text) {
        if (text.isEmpty()) return new int[0];
        BudgetedCharSequence seq = new BudgetedCharSequence(text, source);
        Matcher m = pattern.matcher(seq);
        int[] out = new int[8];
        int n = 0;
        int lastMatchEnd = -1;
        while (find(m, seq, source, text.length())) {
            int ms = m.start();
            int me = m.end();
            if (ms == me && ms == lastMatchEnd) continue;
            if (splitsSurrogatePair(text, ms)) continue;
            if (n + 2 > out.length) out = Arrays.copyOf(out, out.length * 2);
            out[n++] = ms;
            out[n++] = me;
            lastMatchEnd = me;
        }
        return Arrays.copyOf(out, n);
    }

    /** True when {@code index} falls between the two halves of a surrogate pair. */
    private static boolean splitsSurrogatePair(CharSequence s, int index) {
        return index > 0 && index < s.length() && Character.isLowSurrogate(s.charAt(index))
                && Character.isHighSurrogate(s.charAt(index - 1));
    }

    /**
     * The pieces of one word as Hugging Face {@code find_matches} lists them, and their
     * conversion by a split behaviour ({@code NormalizedString::split}).
     */
    private static final class Segments {
        private int[] lengths = new int[16];
        private boolean[] matches = new boolean[16];
        private int size;

        void clear() {
            size = 0;
        }

        void add(int length, boolean match) {
            if (size == lengths.length) {
                lengths = Arrays.copyOf(lengths, size * 2);
                matches = Arrays.copyOf(matches, size * 2);
            }
            lengths[size] = length;
            matches[size] = match;
            size++;
        }

        void emit(Behaviour behaviour, boolean invert, Offsets out) {
            switch (behaviour) {
                case ISOLATED:
                    for (int i = 0; i < size; i++) out.add(lengths[i]);
                    return;
                case REMOVED:
                    for (int i = 0; i < size; i++) {
                        if (matches[i] != invert) out.addRemoved(lengths[i]);
                        else out.add(lengths[i]);
                    }
                    return;
                case CONTIGUOUS:
                case MERGED_WITH_PREVIOUS: {
                    int[] acc = new int[size];
                    int n = 0;
                    boolean previous = false;
                    for (int i = 0; i < size; i++) {
                        boolean match = matches[i] != invert;
                        boolean merge = behaviour == Behaviour.CONTIGUOUS ? match == previous : match && !previous;
                        if (merge && n > 0) acc[n - 1] += lengths[i];
                        else acc[n++] = lengths[i];
                        previous = match;
                    }
                    for (int i = 0; i < n; i++) out.add(acc[i]);
                    return;
                }
                case MERGED_WITH_NEXT: {
                    int[] acc = new int[size];
                    int n = 0;
                    boolean previous = false;
                    for (int i = size - 1; i >= 0; i--) {
                        boolean match = matches[i] != invert;
                        if (match && !previous && n > 0) acc[n - 1] += lengths[i];
                        else acc[n++] = lengths[i];
                        previous = match;
                    }
                    for (int i = n - 1; i >= 0; i--) out.add(acc[i]);
                    return;
                }
                default:
                    throw new IllegalStateException("unhandled split behaviour " + behaviour);
            }
        }
    }

    /** A split at characters selected by a predicate, each one a separate match (HF {@code Fn(char) -> bool}). */
    private static final class PredicateStep implements Step {
        private final IntPredicate predicate;
        private final Behaviour behaviour;

        PredicateStep(IntPredicate predicate, Behaviour behaviour) {
            this.predicate = predicate;
            this.behaviour = behaviour;
        }

        @Override
        public Offsets apply(Text t, Offsets offsets) {
            Offsets out = new Offsets(offsets.size() * 2);
            Segments segments = new Segments();
            int start = 0;
            for (int k = 0; k < offsets.size(); k++) {
                int end = start + offsets.get(k);
                charSegments(t.cps, start, end, predicate, segments);
                segments.emit(behaviour, false, out);
                start = end;
            }
            return out;
        }
    }

    /** The match list of a character predicate over {@code cps[from, to)}. */
    private static void charSegments(int[] cps, int from, int to, IntPredicate predicate, Segments segments) {
        segments.clear();
        int last = from;
        for (int i = from; i < to; i++) {
            if (predicate.test(cps[i])) {
                if (i > last) segments.add(i - last, false);
                segments.add(1, true);
                last = i + 1;
            }
        }
        if (to > last) segments.add(to - last, false);
    }

    // ---------------------------------------------------------------------------------------------
    // Rewriting steps
    // ---------------------------------------------------------------------------------------------

    /** HF ByteLevel {@code add_prefix_space}: a space before every word that does not start with one. */
    private static Offsets prefixSpace(Text t, Offsets offsets) {
        StringBuilder sb = new StringBuilder(t.text.length() + offsets.size());
        Offsets out = new Offsets(offsets.size());
        int start = 0;
        for (int k = 0; k < offsets.size(); k++) {
            int len = offsets.get(k);
            if (len > 0) {
                boolean prefix = t.cps[start] != ' ';
                if (prefix) sb.append(' ');
                sb.append(t.text, t.cpToChar[start], t.cpToChar[start + len]);
                out.add(len + (prefix ? 1 : 0));
            }
            start += len;
        }
        return out.over(new Text(sb.toString(), t.atInputStart));
    }

    /** HF ByteLevel mapping when later steps split the byte-level text: each word GPT-2 byte-encoded. */
    private static Offsets byteLevelEncode(Text t, Offsets offsets) {
        StringBuilder sb = new StringBuilder(t.text.length() * 2);
        Offsets out = new Offsets(offsets.size());
        int start = 0;
        for (int k = 0; k < offsets.size(); k++) {
            int len = offsets.get(k);
            if (len > 0) {
                String encoded = BpeTokenizer.byteEncode(t.text.substring(t.cpToChar[start], t.cpToChar[start + len]));
                sb.append(encoded);
                out.add(encoded.length());
            }
            start += len;
        }
        return out.over(new Text(sb.toString(), t.atInputStart));
    }

    /**
     * HF Metaspace: spaces become the replacement, the replacement is prepended per the scheme,
     * and with {@code split} each word is split before every replacement character
     * ({@code MergedWithNext}).
     */
    private static final class MetaspaceStep implements Step {
        private final int replacement;
        private final PrependScheme scheme;
        private final boolean split;

        MetaspaceStep(int replacement, PrependScheme scheme, boolean split) {
            this.replacement = replacement;
            this.scheme = scheme;
            this.split = split;
        }

        @Override
        public Offsets apply(Text t, Offsets offsets) {
            StringBuilder sb = new StringBuilder(t.text.length() + offsets.size() * 2);
            Offsets out = new Offsets(offsets.size() * 2);
            Segments segments = new Segments();
            boolean first = true;
            int start = 0;
            for (int k = 0; k < offsets.size(); k++) {
                int len = offsets.get(k);
                if (len > 0) {
                    int[] word = rewrite(t.cps, start, start + len, first && t.atInputStart);
                    for (int cp : word) sb.appendCodePoint(cp);
                    if (split) {
                        charSegments(word, 0, word.length, cp -> cp == replacement, segments);
                        segments.emit(Behaviour.MERGED_WITH_NEXT, false, out);
                    } else {
                        out.add(word.length);
                    }
                    first = false;
                }
                start += len;
            }
            return out.over(new Text(sb.toString(), t.atInputStart));
        }

        private int[] rewrite(int[] cps, int from, int to, boolean startsInput) {
            boolean prepend = cps[from] != ' ' && cps[from] != replacement
                    && (scheme == PrependScheme.ALWAYS || (scheme == PrependScheme.FIRST && startsInput));
            int[] word = new int[to - from + (prepend ? 1 : 0)];
            int j = 0;
            if (prepend) word[j++] = replacement;
            for (int i = from; i < to; i++) word[j++] = cps[i] == ' ' ? replacement : cps[i];
            return word;
        }
    }

    /**
     * HF UnicodeScripts: a word is split where the script changes, Hiragana, Katakana and U+30FC
     * count as Han, and a space continues any script. Leading spaces of a word are dropped, since
     * the Hugging Face splits start at the first script change.
     */
    private static Offsets unicodeScripts(Text t, Offsets offsets) {
        Offsets out = new Offsets(offsets.size() * 2);
        int start = 0;
        for (int k = 0; k < offsets.size(); k++) {
            int len = offsets.get(k);
            int end = start + len;
            int last = start;
            boolean opened = false;
            int lastScript = -1;
            for (int i = start; i < end; i++) {
                int script = fixedScript(t.cps[i]);
                if (script == SCRIPT_ANY) continue;
                if (script != lastScript) {
                    if (opened) out.add(i - last);
                    else if (i > start) out.addRemoved(i - start);
                    last = i;
                    opened = true;
                }
                lastScript = script;
            }
            if (opened) out.add(end - last);
            else if (len > 0) out.addRemoved(len);
            start = end;
        }
        return out;
    }

    /** HF {@code fixed_script}: the script ordinal with the Han fusions, or {@link #SCRIPT_ANY} for a space. */
    private static int fixedScript(int cp) {
        if (cp == 0x30FC) return Character.UnicodeScript.HAN.ordinal();
        if (cp == ' ') return SCRIPT_ANY;
        Character.UnicodeScript script = Character.UnicodeScript.of(cp);
        if (script == Character.UnicodeScript.HIRAGANA || script == Character.UnicodeScript.KATAKANA) {
            return Character.UnicodeScript.HAN.ordinal();
        }
        return script.ordinal();
    }

    // ---------------------------------------------------------------------------------------------
    // Character classes of the Hugging Face steps
    // ---------------------------------------------------------------------------------------------

    /** Rust {@code char::is_whitespace}: the Unicode White_Space property. */
    static boolean isWhiteSpace(int cp) {
        if (cp <= 0x20) return cp == 0x20 || (cp >= 0x09 && cp <= 0x0D);
        if (cp < 0x85) return false;
        return cp == 0x85 || cp == 0xA0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028
                || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
    }

    /** Rust {@code char::is_numeric}: general categories Nd, Nl and No. */
    static boolean isNumeric(int cp) {
        if (cp < 0x80) return cp >= '0' && cp <= '9';
        int type = Character.getType(cp);
        return type == Character.DECIMAL_DIGIT_NUMBER || type == Character.LETTER_NUMBER
                || type == Character.OTHER_NUMBER;
    }

    /** HF {@code is_punc}: ASCII punctuation (symbols included) or a Unicode P category. */
    static boolean isPunctuation(int cp) {
        if (cp < 0x80) {
            return (cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96)
                    || (cp >= 123 && cp <= 126);
        }
        switch (Character.getType(cp)) {
            case Character.CONNECTOR_PUNCTUATION:
            case Character.DASH_PUNCTUATION:
            case Character.START_PUNCTUATION:
            case Character.END_PUNCTUATION:
            case Character.INITIAL_QUOTE_PUNCTUATION:
            case Character.FINAL_QUOTE_PUNCTUATION:
            case Character.OTHER_PUNCTUATION:
                return true;
            default:
                return false;
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Oniguruma dialect
    // ---------------------------------------------------------------------------------------------

    /**
     * Compiles a Hugging Face (Oniguruma) regex.
     *
     * @throws IllegalArgumentException when the regex is not valid or uses a construct Java lacks
     */
    static Pattern compileOniguruma(String regex) {
        try {
            return Pattern.compile(onigurumaToJava(regex), ONIGURUMA_FLAGS);
        } catch (PatternSyntaxException e) {
            throw new IllegalArgumentException("Failed to process tokenizer regex '" + regex + "': "
                    + e.getDescription(), e);
        }
    }

    /**
     * Rewrites Oniguruma (Ruby syntax) constructs whose Java meaning differs: property names
     * ({@code \p{Han}}, {@code \p{^L}}, long category names), {@code \h}/{@code \H} (hex digits),
     * {@code \v} (the VT character), POSIX brackets ({@code [[:alpha:]]}), a literal {@code ]}
     * first in a class, {@code (?#...)} comments, and the option letter {@code m} (dot matches a
     * newline, Java's {@code s}). Everything else is shared syntax and is copied.
     */
    static String onigurumaToJava(String re) {
        StringBuilder out = new StringBuilder(re.length() + 16);
        int depth = 0;
        int i = 0;
        while (i < re.length()) {
            char c = re.charAt(i);
            if (c == '\\') {
                if (i + 1 >= re.length()) throw new IllegalArgumentException("Trailing backslash in regex '" + re + "'");
                char e = re.charAt(i + 1);
                if ((e == 'p' || e == 'P') && i + 2 < re.length() && re.charAt(i + 2) == '{') {
                    int close = re.indexOf('}', i + 3);
                    if (close < 0) throw new IllegalArgumentException("Unclosed property name in regex '" + re + "'");
                    String name = re.substring(i + 3, close);
                    boolean negated = e == 'P';
                    if (name.startsWith("^")) {
                        negated = !negated;
                        name = name.substring(1);
                    }
                    out.append(onigProperty(name, negated, re));
                    i = close + 1;
                } else if (e == 'h' || e == 'H') {
                    out.append(e == 'h' ? "[0-9a-fA-F]" : "[^0-9a-fA-F]");
                    i += 2;
                } else if (e == 'v') {
                    out.append("\\x{0B}");
                    i += 2;
                } else {
                    out.append(c).append(e);
                    i += 2;
                }
                continue;
            }
            if (c == '[') {
                if (depth > 0 && i + 1 < re.length() && re.charAt(i + 1) == ':') {
                    int close = re.indexOf(":]", i + 2);
                    if (close > 0) {
                        String name = re.substring(i + 2, close);
                        String posix = posixBracket(name);
                        if (posix != null) {
                            out.append(posix);
                            i = close + 2;
                            continue;
                        }
                    }
                }
                out.append(c);
                depth++;
                i++;
                if (i < re.length() && re.charAt(i) == '^') out.append(re.charAt(i++));
                if (i < re.length() && re.charAt(i) == ']') {
                    out.append("\\]");
                    i++;
                }
                continue;
            }
            if (c == ']' && depth > 0) {
                depth--;
                out.append(c);
                i++;
                continue;
            }
            if (depth == 0 && re.startsWith("(?#", i)) {
                int close = re.indexOf(')', i + 3);
                if (close < 0) throw new IllegalArgumentException("Unclosed comment group in regex '" + re + "'");
                i = close + 1;
                continue;
            }
            if (depth == 0 && re.startsWith("(?", i)) {
                int end = optionGroupEnd(re, i + 2);
                if (end > 0) {
                    out.append("(?").append(re.substring(i + 2, end).replace('m', 's'));
                    i = end;
                    continue;
                }
            }
            out.append(c);
            i++;
        }
        return out.toString();
    }

    /**
     * The index of the {@code )} or {@code :} closing the option letters of a group such as
     * {@code (?i-m)} or {@code (?m:...)} whose letters start at {@code from}, or -1 when the
     * group is not an option group.
     */
    private static int optionGroupEnd(String re, int from) {
        int i = from;
        while (i < re.length() && "imx-".indexOf(re.charAt(i)) >= 0) i++;
        if (i == from || i >= re.length()) return -1;
        char c = re.charAt(i);
        return c == ')' || c == ':' ? i : -1;
    }

    /** The Java form of an Oniguruma {@code \p{name}}. */
    private static String onigProperty(String name, boolean negated, String regex) {
        String key = normalizePropertyName(name);
        if (key.isEmpty()) throw new IllegalArgumentException("Empty property name in regex '" + regex + "'");
        if (key.equals("any")) return negated ? NO_CLASS : ANY_CLASS;
        String body = GENERAL_CATEGORIES.get(key);
        if (body == null) body = POSIX_PROPERTIES.get(key);
        if (body == null) body = scriptProperty(name);
        if (body == null && key.startsWith("in") && key.length() > 2) body = "In" + blockName(name.substring(2));
        if (body == null) body = "Is" + name;
        return (negated ? "\\P{" : "\\p{") + body + "}";
    }

    /**
     * An Oniguruma block name after {@code In} ({@code _Basic_Latin}, {@code BasicLatin}) in a form
     * Java's {@code UnicodeBlock.forName} accepts: leading separators dropped and the others
     * written as underscores, which gives the block's constant name.
     */
    private static String blockName(String rest) {
        int i = 0;
        while (i < rest.length() && (rest.charAt(i) == '_' || rest.charAt(i) == ' ' || rest.charAt(i) == '-')) i++;
        return rest.substring(i).replace(' ', '_').replace('-', '_');
    }

    private static String scriptProperty(String name) {
        try {
            Character.UnicodeScript.forName(name);
            return "Is" + name;
        } catch (IllegalArgumentException e) {
            return null;
        }
    }

    /** The Java class item of a POSIX bracket {@code [:name:]} or {@code [:^name:]}, or null. */
    private static String posixBracket(String name) {
        boolean negated = name.startsWith("^");
        String key = normalizePropertyName(negated ? name.substring(1) : name);
        if (key.equals("word")) return negated ? "\\W" : "\\w";
        String body = POSIX_PROPERTIES.get(key);
        if (body == null) return null;
        return (negated ? "\\P{" : "\\p{") + body + "}";
    }

    /** Oniguruma property-name matching: case, spaces, hyphens and underscores do not matter. */
    private static String normalizePropertyName(String name) {
        StringBuilder sb = new StringBuilder(name.length());
        for (int i = 0; i < name.length(); i++) {
            char c = name.charAt(i);
            if (c != '_' && c != '-' && c != ' ') sb.append(c);
        }
        return sb.toString().toLowerCase(Locale.ROOT);
    }

    private static Map<String, String> generalCategories() {
        String[][] names = {
                {"L", "Letter"}, {"LC", "Cased_Letter"}, {"Lu", "Uppercase_Letter"}, {"Ll", "Lowercase_Letter"},
                {"Lt", "Titlecase_Letter"}, {"Lm", "Modifier_Letter"}, {"Lo", "Other_Letter"},
                {"M", "Mark"}, {"Mn", "Nonspacing_Mark"}, {"Mc", "Spacing_Mark"}, {"Me", "Enclosing_Mark"},
                {"N", "Number"}, {"Nd", "Decimal_Number"}, {"Nl", "Letter_Number"}, {"No", "Other_Number"},
                {"P", "Punctuation"}, {"Pc", "Connector_Punctuation"}, {"Pd", "Dash_Punctuation"},
                {"Ps", "Open_Punctuation"}, {"Pe", "Close_Punctuation"}, {"Pi", "Initial_Punctuation"},
                {"Pf", "Final_Punctuation"}, {"Po", "Other_Punctuation"},
                {"S", "Symbol"}, {"Sm", "Math_Symbol"}, {"Sc", "Currency_Symbol"}, {"Sk", "Modifier_Symbol"},
                {"So", "Other_Symbol"},
                {"Z", "Separator"}, {"Zs", "Space_Separator"}, {"Zl", "Line_Separator"},
                {"Zp", "Paragraph_Separator"},
                {"C", "Other"}, {"Cc", "Control"}, {"Cf", "Format"}, {"Cs", "Surrogate"}, {"Co", "Private_Use"},
                {"Cn", "Unassigned"},
        };
        Map<String, String> m = new HashMap<>();
        for (String[] n : names) {
            m.put(normalizePropertyName(n[0]), n[0]);
            m.put(normalizePropertyName(n[1]), n[0]);
        }
        return Collections.unmodifiableMap(m);
    }

    private static Map<String, String> posixProperties() {
        String[] names = {"Alnum", "Alpha", "ASCII", "Blank", "Cntrl", "Digit", "Graph", "Lower", "Print", "Punct",
                "Space", "Upper", "XDigit"};
        Map<String, String> m = new HashMap<>();
        for (String n : names) m.put(normalizePropertyName(n), n);
        m.put("word", "IsWord");
        return Collections.unmodifiableMap(m);
    }

    // ---------------------------------------------------------------------------------------------
    // llama.cpp regex rewriting
    // ---------------------------------------------------------------------------------------------

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

    /** Every pattern text of one dialect in the {@link PreTokenizerType} table. */
    private static Set<String> builtInRegexes(boolean oniguruma) {
        Set<String> out = new HashSet<>();
        for (PreTokenizerType type : PreTokenizerType.values()) {
            out.addAll(oniguruma ? type.hfRegexes() : type.regexes());
        }
        return Collections.unmodifiableSet(out);
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

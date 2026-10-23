package com.eonmux.neuralfree.gguf.tokenizer;

import com.eonmux.neuralfree.gguf.tokenizer.UnicodeRegexSplit.Offsets;
import com.eonmux.neuralfree.gguf.tokenizer.UnicodeRegexSplit.Step;

import static com.eonmux.neuralfree.gguf.tokenizer.UnicodeFlags.ACCENT_MARK;
import static com.eonmux.neuralfree.gguf.tokenizer.UnicodeFlags.LETTER;
import static com.eonmux.neuralfree.gguf.tokenizer.UnicodeFlags.NUMBER;
import static com.eonmux.neuralfree.gguf.tokenizer.UnicodeFlags.WHITESPACE;

/**
 * The hand-written splitters of llama.cpp {@code unicode_regex_split_custom}
 * (src/unicode.cpp:237-1259), transcribed branch for branch. They are selected by the exact text
 * of the regex they implement, as in llama.cpp; KIMI_K2, AFMOE and the newline-run splitter
 * implement behaviour the trigger text does not spell out.
 */
final class UnicodeCustomSplits {

    /** llama.cpp {@code OUT_OF_RANGE}: the code point read outside the current word. */
    private static final int OUT_OF_RANGE = -1;

    static final String GPT2 = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)";
    static final String LLAMA3_ORIGINAL = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}"
            + "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    static final String LLAMA3 = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+"
            + "|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    static final String QWEN2 = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+"
            + "|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    static final String QWEN35 = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
            + "|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+"
            + "|\\s+(?!\\S)|\\s+";
    static final String K2_HORIZON = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
            + "|[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}|\\p{M}|\\u200C|\\u200D)+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*"
            + "|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    static final String KIMI_K2 = "\\p{Han}+";
    static final String AFMOE_DIGITS = "\\p{AFMoE_digits}";
    static final String NEWLINE_RUNS = "[^\\n]+|[\\n]+";
    static final String TINY_AYA_DIGITS = "\\d{1,3}(?=(?:\\d{3})*\\b)";

    /** Letter-run variants of the llama3-style splitter. */
    private enum Run { LETTERS, LETTERS_MARKS, LETTERS_MARKS_JOINERS }

    private UnicodeCustomSplits() {}

    /** The custom splitter llama.cpp uses for this exact regex text, or null for a generic regex. */
    static Step forRegex(String regex) {
        switch (regex) {
            case GPT2: return (t, o) -> gpt2(t.cps, t.flags, o);
            case LLAMA3_ORIGINAL:
            case LLAMA3: return (t, o) -> llama3Style(t.cps, t.flags, o, Run.LETTERS, 3, false, false);
            case QWEN2: return (t, o) -> llama3Style(t.cps, t.flags, o, Run.LETTERS, 1, false, false);
            case QWEN35: return (t, o) -> llama3Style(t.cps, t.flags, o, Run.LETTERS_MARKS, 1, true, false);
            case K2_HORIZON: return (t, o) -> llama3Style(t.cps, t.flags, o, Run.LETTERS_MARKS_JOINERS, 3, false, true);
            case KIMI_K2: return (t, o) -> kimiK2(t.cps, t.flags, o);
            case AFMOE_DIGITS:
            case TINY_AYA_DIGITS: return (t, o) -> afmoeDigits(t.cps, t.flags, o);
            case NEWLINE_RUNS: return (t, o) -> newlineRuns(t.cps, o);
            default: return null;
        }
    }

    /** The current word: bounded code-point and flag reads plus {@code _add_token}. */
    private static final class Word {
        final int[] cps;
        final int[] flags;
        final int ini;
        final int end;
        final Offsets out;
        int prevEnd;

        Word(int[] cps, int[] flags, int ini, int end, Offsets out) {
            this.cps = cps;
            this.flags = flags;
            this.ini = ini;
            this.end = end;
            this.out = out;
            this.prevEnd = ini;
        }

        int cpt(int pos) { return pos >= ini && pos < end ? cps[pos] : OUT_OF_RANGE; }
        int flags(int pos) { return pos >= ini && pos < end ? flags[pos] : 0; }

        int add(int tokenEnd) {
            int len = tokenEnd - prevEnd;
            if (len > 0) out.add(len);
            prevEnd = tokenEnd;
            return len;
        }
    }

    private static boolean has(int flags, int mask) { return (flags & mask) != 0; }

    /** {@code unicode_regex_split_custom_gpt2} (src/unicode.cpp:237-353). */
    static Offsets gpt2(int[] cps, int[] flags, Offsets offsets) {
        Offsets out = new Offsets(offsets.size());
        int start = 0;
        for (int k = 0; k < offsets.size(); k++) {
            Word w = new Word(cps, flags, start, start + offsets.get(k), out);
            start = w.end;
            for (int pos = w.ini; pos < w.end; ) {
                int cpt = w.cpt(pos);
                int f = w.flags(pos);
                if (cpt == '\'' && pos + 1 < w.end) {
                    int n1 = w.cpt(pos + 1);
                    if (n1 == 's' || n1 == 't' || n1 == 'm' || n1 == 'd') {
                        pos += w.add(pos + 2);
                        continue;
                    }
                    if (pos + 2 < w.end) {
                        int n2 = w.cpt(pos + 2);
                        if ((n1 == 'r' && n2 == 'e') || (n1 == 'v' && n2 == 'e') || (n1 == 'l' && n2 == 'l')) {
                            pos += w.add(pos + 3);
                            continue;
                        }
                    }
                }
                int f2 = cpt == ' ' ? w.flags(pos + 1) : f;
                if (has(f2, LETTER)) {
                    pos += cpt == ' ' ? 1 : 0;
                    while (has(f2, LETTER)) f2 = w.flags(++pos);
                    w.add(pos);
                    continue;
                }
                if (has(f2, NUMBER)) {
                    pos += cpt == ' ' ? 1 : 0;
                    while (has(f2, NUMBER)) f2 = w.flags(++pos);
                    w.add(pos);
                    continue;
                }
                if (!has(f2, WHITESPACE | LETTER | NUMBER) && f2 != 0) {
                    pos += cpt == ' ' ? 1 : 0;
                    while (!has(f2, WHITESPACE | LETTER | NUMBER) && f2 != 0) f2 = w.flags(++pos);
                    w.add(pos);
                    continue;
                }
                pos = whitespace(w, pos, false);
            }
        }
        return out;
    }

    /**
     * {@code unicode_regex_split_custom_llama3} (src/unicode.cpp:355-493) and its variants
     * {@code _qwen2} (643-777: one digit per word), {@code _qwen35} (779-905: letter runs and the
     * punctuation exclusion include marks) and {@code _k2_horizon} (495-641: letter runs include
     * marks, ZWNJ and ZWJ; the contraction folds U+017F to 's').
     */
    static Offsets llama3Style(int[] cps, int[] flags, Offsets offsets, Run run, int maxDigits,
                               boolean punctExcludesMarks, boolean foldLongS) {
        int punctStop = WHITESPACE | LETTER | NUMBER | (punctExcludesMarks ? ACCENT_MARK : 0);
        Offsets out = new Offsets(offsets.size());
        int start = 0;
        for (int k = 0; k < offsets.size(); k++) {
            Word w = new Word(cps, flags, start, start + offsets.get(k), out);
            start = w.end;
            for (int pos = w.ini; pos < w.end; ) {
                int cpt = w.cpt(pos);
                int f = w.flags(pos);
                if (cpt == '\'' && pos + 1 < w.end) {
                    int n1 = UnicodeFlags.toLower(w.cpt(pos + 1));
                    if (foldLongS && n1 == 0x017F) n1 = 's';
                    if (n1 == 's' || n1 == 't' || n1 == 'm' || n1 == 'd') {
                        pos += w.add(pos + 2);
                        continue;
                    }
                    if (pos + 2 < w.end) {
                        int n2 = UnicodeFlags.toLower(w.cpt(pos + 2));
                        if ((n1 == 'r' && n2 == 'e') || (n1 == 'v' && n2 == 'e') || (n1 == 'l' && n2 == 'l')) {
                            pos += w.add(pos + 3);
                            continue;
                        }
                    }
                }
                if (!(cpt == '\r' || cpt == '\n' || has(f, NUMBER))) {
                    if (inRun(w, pos, run) || inRun(w, pos + 1, run)) {
                        pos++;
                        while (inRun(w, pos, run)) pos++;
                        w.add(pos);
                        continue;
                    }
                }
                if (has(f, NUMBER)) {
                    int ini = pos;
                    while (has(w.flags(pos), NUMBER)) {
                        if (++pos - ini >= maxDigits) {
                            w.add(pos);
                            ini = pos;
                        }
                    }
                    w.add(pos);
                    continue;
                }
                int f2 = cpt == ' ' ? w.flags(pos + 1) : f;
                if (!has(f2, punctStop) && f != 0) {
                    pos += cpt == ' ' ? 1 : 0;
                    while (!has(f2, punctStop) && f2 != 0) f2 = w.flags(++pos);
                    int c2 = w.cpt(pos);
                    while (c2 == '\r' || c2 == '\n') c2 = w.cpt(++pos);
                    w.add(pos);
                    continue;
                }
                pos = whitespace(w, pos, true);
            }
        }
        return out;
    }

    private static boolean inRun(Word w, int pos, Run run) {
        int f = w.flags(pos);
        switch (run) {
            case LETTERS: return has(f, LETTER);
            case LETTERS_MARKS: return has(f, LETTER | ACCENT_MARK);
            default:
                int c = w.cpt(pos);
                return c == 0x200C || c == 0x200D || has(f, LETTER | ACCENT_MARK);
        }
    }

    /**
     * The shared tail of the GPT-2 and llama3-style splitters: {@code \s*[\r\n]+} (llama3 style
     * only), {@code \s+(?!\S)}, {@code \s+}, then a single code point. Returns the new position.
     */
    private static int whitespace(Word w, int pos, boolean newlineRuns) {
        int numWhitespaces = 0;
        int lastEndROrN = 0;
        while (has(w.flags(pos + numWhitespaces), WHITESPACE)) {
            int c2 = w.cpt(pos + numWhitespaces);
            if (c2 == '\r' || c2 == '\n') lastEndROrN = pos + numWhitespaces + 1;
            numWhitespaces++;
        }
        if (newlineRuns && lastEndROrN > 0) {
            w.add(lastEndROrN);
            return lastEndROrN;
        }
        if (numWhitespaces > 1 && w.cpt(pos + numWhitespaces) != OUT_OF_RANGE) {
            pos += numWhitespaces - 1;
            w.add(pos);
            return pos;
        }
        if (numWhitespaces > 0) {
            pos += numWhitespaces;
            w.add(pos);
            return pos;
        }
        w.add(++pos);
        return pos;
    }

    /**
     * {@code unicode_regex_split_custom_kimi_k2} (src/unicode.cpp:946-1114): Han runs, letter words
     * excluding Han with optional contractions, digit triples, punctuation runs, then whitespace.
     */
    static Offsets kimiK2(int[] cps, int[] flags, Offsets offsets) {
        Offsets out = new Offsets(offsets.size());
        int start = 0;
        for (int k = 0; k < offsets.size(); k++) {
            Word w = new Word(cps, flags, start, start + offsets.get(k), out);
            start = w.end;
            for (int pos = w.ini; pos < w.end; ) {
                int cpt = w.cpt(pos);
                int f = w.flags(pos);
                if (UnicodeFlags.isHan(cpt)) {
                    while (UnicodeFlags.isHan(w.cpt(pos))) pos++;
                    w.add(pos);
                    continue;
                }
                boolean leadingCandidate = !(cpt == '\r' || cpt == '\n' || has(f, LETTER | NUMBER));
                boolean letterPattern = (has(f, LETTER) && !UnicodeFlags.isHan(cpt))
                        || (leadingCandidate && nonHanLetter(w, pos + 1));
                if (letterPattern) {
                    boolean hasLeadingChar = false;
                    if (leadingCandidate) {
                        hasLeadingChar = true;
                        pos++;
                    }
                    boolean hasLetters = false;
                    while (nonHanLetter(w, pos)) {
                        hasLetters = true;
                        pos++;
                    }
                    if (hasLetters || (!hasLeadingChar && nonHanLetter(w, pos))) {
                        if (!hasLetters) pos++;
                        while (nonHanLetter(w, pos)) pos++;
                        if (w.cpt(pos) == '\'' && pos + 1 < w.end) {
                            int n1 = UnicodeFlags.toLower(w.cpt(pos + 1));
                            if (n1 == 's' || n1 == 't' || n1 == 'm' || n1 == 'd') {
                                pos += 2;
                            } else if (pos + 2 < w.end) {
                                int n2 = UnicodeFlags.toLower(w.cpt(pos + 2));
                                if ((n1 == 'r' && n2 == 'e') || (n1 == 'v' && n2 == 'e') || (n1 == 'l' && n2 == 'l')) {
                                    pos += 3;
                                }
                            }
                        }
                        w.add(pos);
                        continue;
                    } else if (hasLeadingChar) {
                        pos--;
                    }
                }
                if (has(f, NUMBER)) {
                    int ini = pos;
                    while (has(w.flags(pos), NUMBER)) {
                        if (++pos - ini >= 3) {
                            w.add(pos);
                            ini = pos;
                        }
                    }
                    w.add(pos);
                    continue;
                }
                int f2 = cpt == ' ' ? w.flags(pos + 1) : f;
                if (!has(f2, WHITESPACE | LETTER | NUMBER) && f2 != 0) {
                    pos += cpt == ' ' ? 1 : 0;
                    while (!has(f2, WHITESPACE | LETTER | NUMBER) && f2 != 0) f2 = w.flags(++pos);
                    int c2 = w.cpt(pos);
                    while (c2 == '\r' || c2 == '\n') c2 = w.cpt(++pos);
                    w.add(pos);
                    continue;
                }
                pos = whitespace(w, pos, true);
            }
        }
        return out;
    }

    private static boolean nonHanLetter(Word w, int pos) {
        return has(w.flags(pos), LETTER) && !UnicodeFlags.isHan(w.cpt(pos));
    }

    /**
     * {@code unicode_regex_split_custom_afmoe} (src/unicode.cpp:1117-1189): each digit run is split
     * into a leading group of {@code length % 3} digits and then groups of three; the text between
     * digit runs stays one word.
     */
    static Offsets afmoeDigits(int[] cps, int[] flags, Offsets offsets) {
        Offsets out = new Offsets(offsets.size());
        int start = 0;
        for (int k = 0; k < offsets.size(); k++) {
            Word w = new Word(cps, flags, start, start + offsets.get(k), out);
            start = w.end;
            for (int pos = w.ini; pos < w.end; ) {
                if (has(w.flags(pos), NUMBER)) {
                    int digitStart = pos;
                    int digitCount = 0;
                    while (has(w.flags(pos), NUMBER) && pos < w.end) {
                        digitCount++;
                        pos++;
                    }
                    int remainder = digitCount % 3;
                    int current = digitStart;
                    if (remainder > 0) {
                        w.add(current + remainder);
                        current += remainder;
                    }
                    while (current < digitStart + digitCount) {
                        w.add(current + 3);
                        current += 3;
                    }
                    continue;
                }
                pos++;
            }
            if (w.prevEnd < w.end) w.add(w.end);
        }
        return out;
    }

    /** {@code unicode_regex_split_custom_newlines} (src/unicode.cpp:1192-1217): newline and non-newline runs. */
    static Offsets newlineRuns(int[] cps, Offsets offsets) {
        Offsets out = new Offsets(offsets.size());
        int start = 0;
        for (int k = 0; k < offsets.size(); k++) {
            int end = start + offsets.get(k);
            int pos = start;
            while (pos < end) {
                boolean newline = cps[pos] == '\n';
                int runStart = pos;
                while (pos < end && (cps[pos] == '\n') == newline) pos++;
                out.add(pos - runStart);
            }
            start = end;
        }
        return out;
    }
}
